#!/usr/bin/env python3
"""Faithful k=1 Single-Rollout PPO learner with Monte-Carlo prefix credit."""
from __future__ import annotations
import argparse,fcntl,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,'.')
import torch
from peft import PeftModel
from transformers import AutoConfig,AutoModelForCausalLM
from train.online_rl.token_logprobs import completion_logprobs_from_ids
from train.coevolution.single_rollout import business_channels
from train.coevolution.rewards import scalar as inspector_scalar
from train.coevolution.srppo import PrefixCreditHead,business_pass,critic_loss,token_advantages
from train.online_rl.durable_queue import DurableQueue
from train.online_rl.policy_store import PolicyStore
from train.online_rl.schema import OnlineTrajectory

MIN_BATCH={'optimizer':64,'planner':24,'inspector':16}
def channels(t):
 d=t.outcome_after.get('account_delta') or t.outcome_after.get('portfolio_delta')
 if d and all(k in d for k in ('F','C','N')):return business_channels(d)
 recovery=t.outcome_after.get('feedback_recovery')
 if recovery:
  return {'F':inspector_scalar(recovery)*10000.,'C':0.,'N':max(float(recovery['conversions']),0.)*100.,
          'D':max(-float(recovery['cpa_margin']),0.)*10000.}
 return {'F':float(t.reward)*10000,'C':0.,'N':max(float(t.reward)*100,0.),'D':0.}
def prefix_hidden(model,pids,cids,dev):
 ids=torch.tensor([pids+cids],device=dev);out=model.base_model(input_ids=ids,output_hidden_states=True);start=len(pids)-1;return out.hidden_states[-1][0,start:]
def main():
 p=argparse.ArgumentParser();p.add_argument('--role',required=True);p.add_argument('--queue',required=True);p.add_argument('--policy-store',required=True);p.add_argument('--workdir',required=True);p.add_argument('--max-updates',type=int,default=3);p.add_argument('--idle-timeout',type=float,default=7200);p.add_argument('--base',default=None);p.add_argument('--device',default='cuda');p.add_argument('--gpu-lock',default=None);p.add_argument('--actor-lr',type=float,default=1e-6);p.add_argument('--critic-lr',type=float,default=1e-5);p.add_argument('--beta-kl',type=float,default=1e-3);p.add_argument('--seed',type=int,default=30380001);p.add_argument('--min-batch',type=int,default=None);p.add_argument('--thresholds',required=True);a=p.parse_args()
 from train.paths import base_model;a.base=str(base_model(a.base))
 torch.manual_seed(a.seed+{'optimizer':1,'planner':2,'inspector':3}[a.role]);thresholds=json.loads(Path(a.thresholds).read_text())[a.role];q=DurableQueue(a.queue);store=PolicyStore(a.policy_store);work=Path(a.workdir);work.mkdir(parents=True,exist_ok=True);worker=f'srppo-{a.role}';minimum=a.min_batch or MIN_BATCH[a.role];pending=[];updates=0;idle=time.time();logs=[];lock=open(a.gpu_lock or str(work/'gpu.lock'),'a+')
 hidden=int(AutoConfig.from_pretrained(a.base).hidden_size);target=PrefixCreditHead(hidden);critic_path=work/f'srppo_critic_{a.role}.pt'
 if critic_path.exists():target.load_state_dict(torch.load(critic_path,map_location='cpu')['state_dict'])
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
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX);dev=torch.device(a.device);current=int(store.read()['versions'][a.role]);base=AutoModelForCausalLM.from_pretrained(a.base,torch_dtype=torch.bfloat16).to(dev);model=PeftModel.from_pretrained(base,str(store.adapter(a.role,current)),is_trainable=True).to(dev);target=target.to(dev).eval();actor_opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.actor_lr);actor_state=work/f'srppo_actor_opt_{a.role}.pt'
  if actor_state.exists():actor_opt.load_state_dict(torch.load(actor_state,map_location=dev))
  samples=[];labels=[];actor_opt.zero_grad();local_abs=terminal_abs=0.;tokens=0
  loaded_behavior=None
  for msg0 in pending:
   tr=OnlineTrajectory.from_dict(msg0['payload']);bv=int(tr.role_policy_versions[a.role]);nodes=[t for t in tr.transitions if t.role==a.role and t.action_valid and t.completion_token_ids]
   if loaded_behavior!=bv:
    if loaded_behavior is not None:model.delete_adapter('behavior')
    model.load_adapter(str(store.adapter(a.role,bv)),adapter_name='behavior',is_trainable=False);loaded_behavior=bv
   for t in nodes:
    ch=channels(t);label=business_pass(ch,thresholds);labels.append(label)
    with torch.no_grad():h=prefix_hidden(model,t.prompt_token_ids,t.completion_token_ids,dev);qv=torch.sigmoid(target(h));adv=token_advantages(qv,label);local_abs+=float((qv[1:]-qv[:-1]).abs().mean());terminal_abs+=abs(label-float(qv[-1]));tokens+=len(adv)
    model.set_adapter('behavior');old,_=completion_logprobs_from_ids(model,t.prompt_token_ids,t.completion_token_ids,dev);model.set_adapter('default');new,_=completion_logprobs_from_ids(model,t.prompt_token_ids,t.completion_token_ids,dev);ratio=torch.exp(new-old);pg=-(ratio*adv).mean();logr=old-new;kl=(torch.exp(logr)-logr-1).mean();((pg+a.beta_kl*kl)/sum(len([z for z in OnlineTrajectory.from_dict(m['payload']).transitions if z.role==a.role]) for m in pending)).backward();samples.append((t,label));
  if loaded_behavior is not None:model.delete_adapter('behavior')
  actor_opt.step();torch.save(actor_opt.state_dict(),actor_state)
  candidate_head=PrefixCreditHead(hidden).to(dev);candidate_head.load_state_dict(target.state_dict());critic_opt=torch.optim.AdamW(candidate_head.parameters(),lr=a.critic_lr);critic_total=0.
  for _ in range(3):
   critic_opt.zero_grad();losses=[]
   for t,label in samples:
    with torch.no_grad():h=prefix_hidden(model,t.prompt_token_ids,t.completion_token_ids,dev)
    loss,_=critic_loss(candidate_head(h.detach()),label);losses.append(loss)
   loss=sum(losses)/len(losses);loss.backward();critic_opt.step();critic_total=float(loss)
  target=candidate_head.cpu();torch.save({'state_dict':target.state_dict(),'version':updates+1},critic_path);candidate=work/f'{a.role}_candidate'
  if candidate.exists():shutil.rmtree(candidate)
  model.save_pretrained(candidate);rate=sum(labels)/len(labels);metrics={'method':'sr_ppo_k1','samples':len(samples),'positive_rate':rate,'critic_loss':critic_total,'local_delta_abs_mean':local_abs/len(samples),'terminal_residual_abs_mean':terminal_abs/len(samples),'actor_steps':1,'tokens':tokens};store.publish(a.role,str(candidate),current,metrics);updates+=1
  for m in pending:q.ack(m['id'],worker)
  pending=[];logs.append(metrics);del model,base,actor_opt,candidate_head,critic_opt;torch.cuda.empty_cache();fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
 (work/f'learner_{a.role}.json').write_text(json.dumps(logs,indent=2));lock.close()
if __name__=='__main__':main()
