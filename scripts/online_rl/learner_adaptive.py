#!/usr/bin/env python3
"""adaptive prefix-value learner learner with role-normalized advantages and adaptive actor epochs."""
from __future__ import annotations
import argparse, copy, fcntl, json, math, shutil, sys, time
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
from train.online_rl.reward_attribution import adaptive_epoch_count, correlation, explained_variance, role_advantages


def prompt_hidden(model, prompt_ids, device):
    ids=torch.tensor([prompt_ids],device=device)
    with torch.no_grad(): out=model.base_model(input_ids=ids,output_hidden_states=True)
    return out.hidden_states[-1][0,-1].float()


def optimizer_steps(state):
    steps=[]
    for item in state.get('state',{}).values():
        step=item.get('step',0)
        steps.append(int(step.item()) if torch.is_tensor(step) else int(step))
    return max(steps,default=0)


def ratio_diagnostics(model,nodes,old_lps,device,low,high):
    masks=total=0; kls=[]
    with torch.no_grad():
        for t,old_lp in zip(nodes,old_lps):
            new_lp,_=completion_logprobs_from_ids(model,t.prompt_token_ids,t.completion_token_ids,device)
            ratio=torch.exp(new_lp-old_lp); masks+=int(((ratio>=low)&(ratio<=high)).sum());total+=len(ratio)
            logr=old_lp-new_lp;kls.append(float((torch.exp(logr)-logr-1).mean()))
    return masks/max(total,1),sum(kls)/max(len(kls),1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--role',choices=['planner','optimizer','inspector'],required=True)
    ap.add_argument('--queue',required=True);ap.add_argument('--policy-store',required=True);ap.add_argument('--workdir',required=True)
    ap.add_argument('--worker-id',default=None);ap.add_argument('--max-updates',type=int,default=2);ap.add_argument('--max-lag',type=int,default=2)
    ap.add_argument('--lr',type=float,default=1e-6);ap.add_argument('--value-lr',type=float,default=1e-3);ap.add_argument('--gamma',type=float,default=1.0);ap.add_argument('--gae-lambda',type=float,default=.95)
    ap.add_argument('--ratio-low',type=float,default=.8);ap.add_argument('--ratio-high',type=float,default=1.2);ap.add_argument('--beta-kl',type=float,default=.05)
    ap.add_argument('--actor-epochs',type=int,default=4);ap.add_argument('--target-kl',type=float,default=.01);ap.add_argument('--min-mask-frac',type=float,default=.4)
    ap.add_argument('--critic-warmup-updates',type=int,default=1)
    ap.add_argument('--base',default=None);ap.add_argument('--idle-timeout',type=float,default=7200)
    ap.add_argument('--gpu-lock',default=None);ap.add_argument('--device',default='cuda');ap.add_argument('--seed',type=int,default=20270001)
    args=ap.parse_args();worker=args.worker_id or f'prefix-value-{args.role}'
    from train.paths import base_model
    args.base=str(base_model(args.base))
    torch.manual_seed(args.seed+{'optimizer':1,'planner':2,'inspector':3}[args.role])
    q=DurableQueue(args.queue);store=PolicyStore(args.policy_store);work=Path(args.workdir);work.mkdir(parents=True,exist_ok=True)
    hidden_size=int(AutoConfig.from_pretrained(args.base).hidden_size);value=PrefixValueModel(hidden_size)
    state_path=work/f'adaptive_value_state_{args.role}.pt';persistent=torch.load(state_path,map_location='cpu') if state_path.exists() else {}
    if persistent.get('value_state'):value.load_state_dict(persistent['value_state'])
    completed=int(persistent.get('completed_updates',0));updates=0;idle_since=time.time();log=[]
    lock_path=args.gpu_lock or str(work/'learner_gpu.lock');Path(lock_path).parent.mkdir(parents=True,exist_ok=True);lock_file=open(lock_path,'a+')
    while updates<args.max_updates:
        msg=q.claim(args.role,worker)
        if msg is None:
            if time.time()-idle_since>args.idle_timeout:break
            time.sleep(.5);continue
        idle_since=time.time()
        try:
            traj=OnlineTrajectory.from_dict(msg['payload']);behavior_version=int(traj.role_policy_versions[args.role]);current=int(store.read()['versions'][args.role]);lag=current-behavior_version
            if lag<0 or lag>args.max_lag:q.ack(msg['id'],worker);log.append({'trajectory':traj.trajectory_id,'skipped':'lag','lag':lag});continue
            nodes=[t for t in traj.transitions if t.role==args.role and t.action_valid and t.completion_token_ids]
            if not nodes:
                fcntl.flock(lock_file.fileno(),fcntl.LOCK_EX)
                try:
                    current=int(store.read()['versions'][args.role])
                    new_version=store.publish(args.role,str(store.adapter(args.role,current)),current,
                                              {'skipped':'no_valid_nodes','trajectory':traj.trajectory_id})
                finally:
                    fcntl.flock(lock_file.fileno(),fcntl.LOCK_UN)
                updates+=1;q.ack(msg['id'],worker)
                log.append({'version':new_version,'trajectory':traj.trajectory_id,
                            'skipped':'no_valid_nodes'})
                continue
            fcntl.flock(lock_file.fileno(),fcntl.LOCK_EX);device=torch.device(args.device)
            base=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.bfloat16).to(device)
            model=PeftModel.from_pretrained(base,str(store.adapter(args.role,current)),is_trainable=True).to(device)
            actor_opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
            value=value.to(device);value_opt=torch.optim.AdamW(value.parameters(),lr=args.value_lr)
            if persistent.get('actor_optimizer'):actor_opt.load_state_dict(persistent['actor_optimizer'])
            if persistent.get('value_optimizer'):value_opt.load_state_dict(persistent['value_optimizer'])
            actor_step_before=optimizer_steps(actor_opt.state_dict());value_step_before=optimizer_steps(value_opt.state_dict())
            h=torch.stack([prompt_hidden(model,t.prompt_token_ids,device) for t in nodes])
            with torch.no_grad():vals=value(h).cpu().tolist()
            behavior_name='prefix_behavior';model.load_adapter(str(store.adapter(args.role,behavior_version)),adapter_name=behavior_name,is_trainable=False);model.set_adapter(behavior_name)
            old_lps=[]
            with torch.no_grad():
                for t in nodes:
                    lp,_=completion_logprobs_from_ids(model,t.prompt_token_ids,t.completion_token_ids,device);old_lps.append(lp.detach())
            model.set_adapter('default');model.delete_adapter(behavior_name)
            rewards=[t.reward for t in nodes]
            gae=account_isolated_gae(nodes,vals,args.gamma,args.gae_lambda)
            inspector_baseline=float(persistent.get('inspector_reward_baseline',0.0))
            raw_advantages=list(gae.advantages)
            advantages=role_advantages(args.role,raw_advantages,rewards,inspector_baseline)
            epoch_budget=adaptive_epoch_count(advantages,args.actor_epochs)
            targets=torch.tensor(gae.returns,device=device);pred=value(h.detach())
            value_loss=F.smooth_l1_loss(pred,targets);value_opt.zero_grad();value_loss.backward();value_opt.step()
            warmup=args.critic_warmup_updates if args.role=='optimizer' else 0
            critic_only=completed<warmup;epoch_logs=[];accepted_epochs=0
            if not critic_only:
                for epoch in range(epoch_budget):
                    pre_mask,pre_kl=ratio_diagnostics(model,nodes,old_lps,device,args.ratio_low,args.ratio_high)
                    if pre_kl>args.target_kl or pre_mask<args.min_mask_frac:
                        epoch_logs.append({'epoch':epoch,'accepted':False,'reason':'pre_gate','mask_frac':pre_mask,'kl':pre_kl});break
                    param_snapshot={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad};opt_snapshot=copy.deepcopy(actor_opt.state_dict())
                    actor_opt.zero_grad();loss_sum=0.0
                    for t,adv,old_lp in zip(nodes,advantages,old_lps):
                        new_lp,_=completion_logprobs_from_ids(model,t.prompt_token_ids,t.completion_token_ids,device)
                        pg,_,_=policy_loss('token_ratio_mask',new_lp,old_lp,float(adv),args.ratio_low,args.ratio_high)
                        logr=old_lp-new_lp;kl=(torch.exp(logr)-logr-1).mean();loss=(pg+args.beta_kl*kl)/len(nodes);loss.backward();loss_sum+=float(loss.detach())
                    actor_opt.step();post_mask,post_kl=ratio_diagnostics(model,nodes,old_lps,device,args.ratio_low,args.ratio_high)
                    if post_kl>args.target_kl or post_mask<args.min_mask_frac:
                        with torch.no_grad():
                            for n,p in model.named_parameters():
                                if p.requires_grad:p.copy_(param_snapshot[n].to(device=p.device,dtype=p.dtype))
                        actor_opt.load_state_dict(opt_snapshot);epoch_logs.append({'epoch':epoch,'accepted':False,'reason':'post_gate','mask_frac':post_mask,'kl':post_kl,'loss':loss_sum});break
                    accepted_epochs+=1;epoch_logs.append({'epoch':epoch,'accepted':True,'mask_frac':post_mask,'kl':post_kl,'loss':loss_sum})
            candidate=work/f'{args.role}_candidate'
            if candidate.exists():shutil.rmtree(candidate)
            model.save_pretrained(candidate);completed+=1
            updated_inspector_baseline=(0.9*inspector_baseline+0.1*sum(rewards)/len(rewards)) if args.role=='inspector' else inspector_baseline
            persistent={'implementation_version':'adaptive-prefix-value','completed_updates':completed,'value_state':{k:v.detach().cpu() for k,v in value.state_dict().items()},'actor_optimizer':actor_opt.state_dict(),'value_optimizer':value_opt.state_dict(),'inspector_reward_baseline':updated_inspector_baseline}
            torch.save(persistent,state_path)
            pred_cpu=pred.detach().cpu().tolist();ret=list(gae.returns);adv=advantages
            raw=raw_advantages;norm=advantages
            metrics={'implementation_version':'adaptive-prefix-value','trajectory':traj.trajectory_id,'lag':lag,'critic_only':critic_only,'accepted_actor_epochs':accepted_epochs,'adaptive_epoch_budget':epoch_budget,'epochs':epoch_logs,'value_loss':float(value_loss.detach()),'explained_variance':explained_variance(pred_cpu,ret),'value_return_correlation':correlation(pred_cpu,ret),'raw_advantage_mean':sum(raw)/len(raw),'raw_advantage_std':math.sqrt(sum((x-sum(raw)/len(raw))**2 for x in raw)/len(raw)),'advantage_mean':sum(norm)/len(norm),'advantage_std':math.sqrt(sum((x-sum(norm)/len(norm))**2 for x in norm)/len(norm)),'positive_advantage_fraction':sum(x>0 for x in norm)/len(norm),'inspector_baseline_before':inspector_baseline,'inspector_baseline_after':updated_inspector_baseline,'actor_optimizer_step_before':actor_step_before,'actor_optimizer_step_after':optimizer_steps(actor_opt.state_dict()),'value_optimizer_step_before':value_step_before,'value_optimizer_step_after':optimizer_steps(value_opt.state_dict()),'actor_epochs_requested':args.actor_epochs,'target_kl':args.target_kl,'min_mask_frac':args.min_mask_frac}
            new_version=store.publish(args.role,str(candidate),current,metrics);updates+=1;q.ack(msg['id'],worker);log.append({'version':new_version,**metrics})
            value=value.cpu();del model,base,actor_opt,value_opt,h,pred,targets,old_lps;torch.cuda.empty_cache()
            persistent=torch.load(state_path,map_location='cpu');fcntl.flock(lock_file.fileno(),fcntl.LOCK_UN)
        except Exception as exc:
            try:fcntl.flock(lock_file.fileno(),fcntl.LOCK_UN)
            except:pass
            q.release(msg['id'],worker);log.append({'message_id':msg['id'],'error':f'{type(exc).__name__}: {exc}'});raise
    (work/f'learner_{args.role}.json').write_text(json.dumps(log,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8');lock_file.close()

if __name__=='__main__':main()
