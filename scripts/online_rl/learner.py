#!/usr/bin/env python3
"""Asynchronous role learner for complete single-rollout AdCoEvo.

Each queued update acquires the GPU lease, loads the latest role policy, updates the
value model and actor, publishes a new version, then releases GPU memory. Role learners
consume the scheduled, settled records produced by the collector.
"""
from __future__ import annotations
import argparse, fcntl, json, shutil, sys, time
from pathlib import Path
sys.path.insert(0,'.')
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM
from train.online_rl.token_logprobs import completion_logprobs_from_ids
from train.online_rl.durable_queue import DurableQueue
from train.online_rl.policy_store import PolicyStore
from train.online_rl.schema import OnlineTrajectory
from train.online_rl.value import PrefixValueModel, account_isolated_gae
from train.online_rl.surrogate import policy_loss


def hidden_for(model,pids,cids,dev):
    ids=torch.tensor([pids+cids],device=dev)
    with torch.no_grad(): out=model.base_model(input_ids=ids,output_hidden_states=True)
    return out.hidden_states[-1][0,-1].float()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--role',choices=['planner','optimizer','inspector'],required=True)
    ap.add_argument('--queue',required=True); ap.add_argument('--policy-store',required=True); ap.add_argument('--workdir',required=True)
    ap.add_argument('--worker-id',default=None); ap.add_argument('--max-updates',type=int,default=2); ap.add_argument('--max-lag',type=int,default=2)
    ap.add_argument('--lr',type=float,default=1e-6); ap.add_argument('--value-lr',type=float,default=1e-3); ap.add_argument('--gamma',type=float,default=1.0); ap.add_argument('--gae-lambda',type=float,default=.95)
    ap.add_argument('--ratio-low',type=float,default=.8); ap.add_argument('--ratio-high',type=float,default=1.2); ap.add_argument('--beta-kl',type=float,default=.05)
    ap.add_argument('--surrogate',choices=['ppo_clip','token_ratio_mask','cispo_clipped_is','gspo_sequence_clip','sapo_soft_gate','dapo_asymmetric_clip'],default='token_ratio_mask')
    ap.add_argument('--base',default=None); ap.add_argument('--idle-timeout',type=float,default=120)
    ap.add_argument('--gpu-lock',default=None)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--seed',type=int,default=20270001)
    args=ap.parse_args(); worker=args.worker_id or f'learner-{args.role}'
    from train.paths import base_model
    args.base=str(base_model(args.base))
    torch.manual_seed(args.seed + {"optimizer":1,"planner":2,"inspector":3}[args.role])
    q=DurableQueue(args.queue); store=PolicyStore(args.policy_store); work=Path(args.workdir); work.mkdir(parents=True,exist_ok=True)
    hidden_size=int(AutoConfig.from_pretrained(args.base).hidden_size)
    value=PrefixValueModel(hidden_size); value_path=work/f'value_{args.role}.pt'
    if value_path.exists(): value.load_state_dict(torch.load(value_path,map_location='cpu')['state_dict'])
    updates=0; idle_since=time.time(); log=[]
    lock_path=args.gpu_lock or str(work/'learner_gpu.lock'); Path(lock_path).parent.mkdir(parents=True,exist_ok=True)
    lock_file=open(lock_path,'a+')
    while updates<args.max_updates:
        msg=q.claim(args.role,worker)
        if msg is None:
            if time.time()-idle_since>args.idle_timeout: break
            time.sleep(.5); continue
        idle_since=time.time()
        try:
            traj=OnlineTrajectory.from_dict(msg['payload']); behavior_version=int(traj.role_policy_versions[args.role])
            current=int(store.read()['versions'][args.role]); lag=current-behavior_version
            if lag<0 or lag>args.max_lag:
                q.ack(msg['id'],worker); log.append({'trajectory':traj.trajectory_id,'skipped':'lag','lag':lag}); continue
            nodes=[t for t in traj.transitions if t.role==args.role and t.action_valid and t.completion_token_ids]
            if not nodes:
                fcntl.flock(lock_file.fileno(),fcntl.LOCK_EX)
                try:
                    current=int(store.read()['versions'][args.role])
                    new_version=store.publish(args.role,str(store.adapter(args.role,current)),current,
                                              {'skipped':'no_valid_nodes','trajectory':traj.trajectory_id})
                finally:
                    fcntl.flock(lock_file.fileno(),fcntl.LOCK_UN)
                updates+=1; q.ack(msg['id'],worker)
                log.append({'version':new_version,'trajectory':traj.trajectory_id,
                            'skipped':'no_valid_nodes'})
                continue
            print(f'[learner:{args.role}] waiting GPU for trajectory={traj.trajectory_id}',flush=True)
            fcntl.flock(lock_file.fileno(),fcntl.LOCK_EX)
            dev=torch.device(args.device)
            base=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.bfloat16).to(dev)
            model=PeftModel.from_pretrained(base,str(store.adapter(args.role,current)),is_trainable=True).to(dev)
            actor_opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
            value=value.to(dev); value_opt=torch.optim.AdamW(value.parameters(),lr=args.value_lr)
            h=torch.stack([hidden_for(model,t.prompt_token_ids,t.completion_token_ids,dev) for t in nodes])
            with torch.no_grad(): vals=value(h).cpu().tolist()
            behavior_name='online_behavior'
            model.load_adapter(str(store.adapter(args.role,behavior_version)),adapter_name=behavior_name,is_trainable=False)
            model.set_adapter(behavior_name); behavior_lps=[]
            with torch.no_grad():
                for t in nodes:
                    lp,_=completion_logprobs_from_ids(model,t.prompt_token_ids,t.completion_token_ids,dev); behavior_lps.append(lp.detach())
            model.set_adapter('default'); model.delete_adapter(behavior_name)
            gae=account_isolated_gae(nodes,vals,args.gamma,args.gae_lambda)
            advantages=list(gae.advantages)
            pred=value(h.detach()); target=torch.tensor(gae.returns,device=dev)
            vloss=F.smooth_l1_loss(pred,target); value_opt.zero_grad(); vloss.backward(); value_opt.step()
            actor_opt.zero_grad(); loss_sum=0.; mask_tokens=total_tokens=0; kl_sum=0.
            for t,adv,old_lp in zip(nodes,advantages,behavior_lps):
                new_lp,_=completion_logprobs_from_ids(model,t.prompt_token_ids,t.completion_token_ids,dev)
                pg,ratio,mask=policy_loss(args.surrogate,new_lp,old_lp,float(adv),args.ratio_low,args.ratio_high)
                logr=old_lp-new_lp; kl=(torch.exp(logr)-logr-1).mean(); action_loss=(pg+args.beta_kl*kl)/len(nodes)
                action_loss.backward(); loss_sum+=float(action_loss.detach()); mask_tokens+=int(mask.sum().item()); total_tokens+=len(mask); kl_sum+=float(kl.detach())
                del new_lp,ratio,mask,pg,kl,action_loss
            actor_opt.step(); candidate=work/f'{args.role}_candidate'
            if candidate.exists(): shutil.rmtree(candidate)
            model.save_pretrained(candidate); value=value.cpu(); torch.save({'state_dict':value.state_dict(),'version':current},value_path)
            metrics={'trajectory':traj.trajectory_id,'lag':lag,'actor_loss':loss_sum,'value_loss':float(vloss.detach()),'mask_frac':mask_tokens/max(total_tokens,1),'kl':kl_sum/len(nodes),'surrogate':args.surrogate,'advantage_estimator':'gae'}
            new_version=store.publish(args.role,str(candidate),current,metrics); updates+=1; q.ack(msg['id'],worker)
            log.append({'trajectory':traj.trajectory_id,'version':new_version,**metrics})
            print(f'[learner:{args.role}] update={updates} version={new_version} lag={lag} mask={metrics["mask_frac"]:.3f}',flush=True)
            del model,base,actor_opt,value_opt,h,pred,target,behavior_lps; torch.cuda.empty_cache(); fcntl.flock(lock_file.fileno(),fcntl.LOCK_UN)
        except Exception as e:
            try: fcntl.flock(lock_file.fileno(),fcntl.LOCK_UN)
            except: pass
            q.release(msg['id'],worker); log.append({'message_id':msg['id'],'error':f'{type(e).__name__}: {e}'}); raise
    (work/f'learner_{args.role}.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    lock_file.close(); print(f'[learner:{args.role}] done updates={updates} queue={q.stats()}')
if __name__=='__main__': main()
