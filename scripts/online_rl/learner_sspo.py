#!/usr/bin/env python3
"""Continuous Single-stream Policy Optimization learner."""
from __future__ import annotations
import argparse,fcntl,json,math,shutil,sys,time
from pathlib import Path
sys.path.insert(0,'.')
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from train.online_rl.token_logprobs import completion_logprobs_from_ids
from train.coevolution.single_rollout import EMATracker,business_channels,business_reward,normalize,tracker_key
from train.coevolution.rewards import scalar as inspector_scalar
from train.online_rl.durable_queue import DurableQueue
from train.online_rl.policy_store import PolicyStore
from train.online_rl.schema import OnlineTrajectory

MIN_BATCH={'optimizer':64,'planner':24,'inspector':16}

def outcome_channels(t):
 o=t.outcome_after
 d=o.get('account_delta') or o.get('feedback_recovery') or o.get('portfolio_delta')
 if d and all(k in d for k in ('F','C','N')):return business_channels(d)
 if d and 'adv' in d:return {'F':inspector_scalar(d)*10000,'C':0.,'N':0.,'D':0.}
 return {'F':float(t.reward)*10000,'C':0.,'N':0.,'D':0.}

def main():
 p=argparse.ArgumentParser();p.add_argument('--role',required=True);p.add_argument('--queue',required=True);p.add_argument('--policy-store',required=True);p.add_argument('--workdir',required=True);p.add_argument('--max-updates',type=int,default=3);p.add_argument('--idle-timeout',type=float,default=7200);p.add_argument('--base',default=None);p.add_argument('--device',default='cuda');p.add_argument('--gpu-lock',default=None);p.add_argument('--lr',type=float,default=1e-6);p.add_argument('--beta-kl',type=float,default=1e-3);p.add_argument('--seed',type=int,default=30380001);p.add_argument('--min-batch',type=int,default=None);a=p.parse_args()
 from train.paths import base_model;a.base=str(base_model(a.base))
 torch.manual_seed(a.seed+{'optimizer':1,'planner':2,'inspector':3}[a.role]);q=DurableQueue(a.queue);store=PolicyStore(a.policy_store);work=Path(a.workdir);work.mkdir(parents=True,exist_ok=True);tracker=EMATracker(work/f'sspo_tracker_{a.role}.json');minimum=a.min_batch or MIN_BATCH[a.role];worker=f'sspo-{a.role}';pending=[];updates=0;idle=time.time();logs=[]
 lock=open(a.gpu_lock or str(work/'gpu.lock'),'a+')
 while updates<a.max_updates:
  msg=q.claim(a.role,worker)
  if msg is not None:
   tr=OnlineTrajectory.from_dict(msg['payload'])
   if not any(t.role==a.role and t.action_valid and t.completion_token_ids for t in tr.transitions):
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    try:
     current=int(store.read()['versions'][a.role])
     version=store.publish(a.role,str(store.adapter(a.role,current)),current,
                           {'skipped':'no_valid_nodes','trajectory':tr.trajectory_id})
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    updates+=1;q.ack(msg['id'],worker);logs.append({'version':version,'skipped':'no_valid_nodes'});idle=time.time();continue
   pending.append(msg);idle=time.time()
  sample_count=sum(sum(1 for t in OnlineTrajectory.from_dict(m['payload']).transitions if t.role==a.role and t.action_valid and t.completion_token_ids) for m in pending)
  if sample_count<minimum:
   if time.time()-idle>a.idle_timeout:break
   time.sleep(.2);continue
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX);dev=torch.device(a.device);current=int(store.read()['versions'][a.role]);base=AutoModelForCausalLM.from_pretrained(a.base,torch_dtype=torch.bfloat16).to(dev);model=PeftModel.from_pretrained(base,str(store.adapter(a.role,current)),is_trainable=True).to(dev);opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.lr)
  state_path=work/f'sspo_optimizer_{a.role}.pt'
  if state_path.exists():opt.load_state_dict(torch.load(state_path,map_location=dev))
  samples=[];tracker_updates=[];first=0
  for msg0 in pending:
   tr=OnlineTrajectory.from_dict(msg0['payload']);bv=int(tr.role_policy_versions[a.role]);model.load_adapter(str(store.adapter(a.role,bv)),adapter_name='behavior',is_trainable=False);nodes=[t for t in tr.transitions if t.role==a.role and t.action_valid and t.completion_token_ids]
   for t in nodes:
    ch=outcome_channels(t);r=business_reward(ch);keys=tracker_key(t);v,source_key,level,is_first=tracker.lookup(keys);first+=is_first;update_key=keys[0];samples.append((t,r-v,r,bv,update_key,level));tracker_updates.append((update_key,r,t))
   model.delete_adapter('behavior')
  adv,ok=normalize([x[1] for x in samples]);loss_value=0.;kl_value=0.
  if ok:
   opt.zero_grad()
   for (t,_,_,bv,_,_),ad in zip(samples,adv):
    model.load_adapter(str(store.adapter(a.role,bv)),adapter_name='behavior',is_trainable=False);model.set_adapter('behavior')
    with torch.no_grad():old,_=completion_logprobs_from_ids(model,t.prompt_token_ids,t.completion_token_ids,dev)
    model.set_adapter('default');new,_=completion_logprobs_from_ids(model,t.prompt_token_ids,t.completion_token_ids,dev);ratio=torch.exp(new-old);s1=ratio*ad;s2=torch.clamp(ratio,.8,1.28)*ad;pg=-torch.minimum(s1,s2).mean();logr=old-new;kl=(torch.exp(logr)-logr-1).mean();loss=(pg+a.beta_kl*kl)/len(samples);loss.backward();loss_value+=float(loss);kl_value+=float(kl);model.delete_adapter('behavior')
   opt.step()
  for key,r,t in tracker_updates:
   state=tracker.states.get(key);drift=0.
   if state and state.last_prompt and state.last_tokens and state.last_logprobs:
    with torch.no_grad():cur,_=completion_logprobs_from_ids(model,state.last_prompt,state.last_tokens,dev)
    old=torch.tensor(state.last_logprobs,device=dev);drift=float((cur-old).abs().mean())
   tracker.update(key,r,tracker.rho(drift),t.prompt_token_ids,t.completion_token_ids,t.behavior_token_logprobs)
  tracker.save();torch.save(opt.state_dict(),state_path);candidate=work/f'{a.role}_candidate'
  if candidate.exists():shutil.rmtree(candidate)
  model.save_pretrained(candidate);metrics={'method':'sspo_continuous','samples':len(samples),'actor_updated':ok,'adv_mean':sum(adv)/len(adv),'adv_std':math.sqrt(sum(x*x for x in adv)/len(adv)),'first_seen_ratio':first/len(samples),'tracker_keys':len(tracker.states),'loss':loss_value,'kl':kl_value/max(len(samples),1)};store.publish(a.role,str(candidate),current,metrics);updates+=1
  for m in pending:q.ack(m['id'],worker)
  pending=[];logs.append(metrics);del model,base,opt;torch.cuda.empty_cache();fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
 (work/f'learner_{a.role}.json').write_text(json.dumps(logs,indent=2));lock.close()
if __name__=='__main__':main()
