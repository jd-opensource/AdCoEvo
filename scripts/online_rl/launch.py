#!/usr/bin/env python3
"""Launch complete asynchronous AdCoEvo collector + three role learners locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.online_rl.policy_store import PolicyStore
from train.coevolution.ablation import MODES
from train.coevolution.rewards import validate_stats
from train.online_rl.budget_regime import ROLE_SCHEDULES, scheduled_update_counts


def parse_episodes(value: str) -> list[int]:
    if '-' in value:
        start, end = (int(part) for part in value.split('-', 1))
        episodes = list(range(start, end + 1))
    else:
        episodes = [int(part) for part in value.split(',')]
    if not episodes or len(set(episodes)) != len(episodes) or episodes != sorted(episodes):
        raise ValueError('episodes must be a nonempty, increasing sequence without duplicates')
    return episodes


def popen(cmd, log, env=None):
    f=open(log,'w',encoding='utf-8')
    return subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,text=True,env=env),f


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',default='runs/online_async_r0'); ap.add_argument('--episodes',default='0-2')
    ap.add_argument('--pv',type=int,default=500000); ap.add_argument('--updates',type=int,default=2)
    ap.add_argument('--K',type=int,default=8,help='maximum tCPA changes per unit and period')
    ap.add_argument('--window',type=int,default=12,help='warm-up ticks W: first W ticks run native budget + all HOLD, agent decisions start at tick W')
    ap.add_argument('--budget-scale',type=float,default=1.0); ap.add_argument('--collector-out',default=None)
    ap.add_argument('--env-profile',choices=['general','sparse','extra_sparse_feedback_stress'],default='sparse')
    ap.add_argument('--inspector-stats',default=None,
                    help='JSON with frozen training-set ADV/conversion means and standard deviations')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--temperature',type=float,default=0.7)
    ap.add_argument('--schedule-origin',type=int,default=None); ap.add_argument('--role-schedule',choices=list(ROLE_SCHEDULES),default='round_robin')
    ap.add_argument('--optimizer-updates',type=int,default=None); ap.add_argument('--planner-updates',type=int,default=None); ap.add_argument('--inspector-updates',type=int,default=None)
    ap.add_argument('--surrogate',default='token_ratio_mask')
    ap.add_argument('--wait-for-role-update',action='store_true')
    ap.add_argument('--ablation-mode',choices=MODES,default='full')
    ap.add_argument('--prefix-value',action='store_true')
    ap.add_argument('--adaptive-prefix-value',action='store_true')
    ap.add_argument('--actor-epochs',type=int,default=4)
    ap.add_argument('--target-kl',type=float,default=.01)
    ap.add_argument('--min-mask-frac',type=float,default=.4)
    ap.add_argument('--critic-warmup-updates',type=int,default=1)
    ap.add_argument('--portfolio-reward-weight',type=float,default=.2)
    ap.add_argument('--learner-idle-timeout',type=float,default=7200)
    ap.add_argument('--server-startup-timeout',type=float,default=1200)
    ap.add_argument('--learner-lr',type=float,default=1e-6)
    ap.add_argument('--value-lr',type=float,default=1e-3)
    ap.add_argument('--beta-kl',type=float,default=None)
    # Explicit learner routing: lets the unified entry feed SPO/SR-PPO style learners
    # (different flag surfaces) without duplicating process supervision here.
    ap.add_argument('--learner-script',default=None,help='override learner script for every role')
    ap.add_argument('--learner-arg',action='append',default=[],help='extra learner flag, repeatable; use key=value')
    ap.add_argument('--base',default=None)
    ap.add_argument('--served-model-name',default=None,help='identifier the local vLLM server exposes; any base model works (env ADCOEVO_SERVED_MODEL, default "model")')
    ap.add_argument('--planner-init',default=None); ap.add_argument('--optimizer-init',default=None); ap.add_argument('--inspector-init',default=None)
    ap.add_argument('--port',type=int,default=8001)
    ap.add_argument('--seed',type=int,default=20270001)
    ap.add_argument('--rollout-gpu',type=int,default=0)
    ap.add_argument('--optimizer-gpu',type=int,default=None)
    ap.add_argument('--planner-gpu',type=int,default=None)
    ap.add_argument('--inspector-gpu',type=int,default=None)
    args=ap.parse_args()
    if args.prefix_value and not args.adaptive_prefix_value and args.beta_kl is not None:
        raise SystemExit('--beta-kl is not used by prefix-value learner; KL is a monitoring and rollback gate')
    if args.beta_kl is None:
        args.beta_kl=.05
    try:
        episodes=parse_episodes(args.episodes)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.schedule_origin is None:
        args.schedule_origin=episodes[0]
    if not args.inspector_stats or not Path(args.inspector_stats).is_file():
        raise SystemExit('--inspector-stats must name a JSON file of frozen training-set statistics')
    try:
        validate_stats(json.loads(Path(args.inspector_stats).read_text(encoding='utf-8')))
    except (ValueError, TypeError) as exc:
        raise SystemExit(f'invalid --inspector-stats: {exc}')
    if not args.resume:
        for role in ('planner','optimizer','inspector'):
            adapter=getattr(args,f'{role}_init')
            if not adapter or not Path(adapter).is_dir():
                raise SystemExit(f'--{role}-init must name an existing adapter directory')
    if args.wait_for_role_update:
        target_updates=scheduled_update_counts(episodes,args.schedule_origin,args.role_schedule,
                                               args.ablation_mode)
        for role in target_updates:
            override=getattr(args,f'{role}_updates')
            if override is not None and override != target_updates[role]:
                raise SystemExit(f'--{role}-updates must equal {target_updates[role]} for this '
                                 'wait-for-update schedule')
    else:
        target_updates={role:(getattr(args,f'{role}_updates') if getattr(args,f'{role}_updates') is not None
                              else args.updates) for role in ('planner','optimizer','inspector')}
    root=Path(args.run_dir); (root/'logs').mkdir(parents=True,exist_ok=True)
    from train.paths import base_model, served_model_name
    args.base=str(base_model(args.base))
    args.served_model_name=served_model_name(args.served_model_name)
    store=PolicyStore(str(root/'policies'))
    if args.resume:
        if not store.manifest.exists(): raise SystemExit('cannot resume: PolicyStore manifest missing')
    else:
        store.initialize({'planner':args.planner_init,'optimizer':args.optimizer_init,'inspector':args.inspector_init})
    queue=str(root/'queue.sqlite'); py=sys.executable
    # Initialize SQLite/WAL once in the parent before concurrent learner startup.
    from train.online_rl.durable_queue import DurableQueue
    durable_queue=DurableQueue(queue)
    if args.resume and args.collector_out:
        pending_path=Path(args.collector_out)/'pending_episode.json'
        if pending_path.is_file():
            pending=json.loads(pending_path.read_text(encoding='utf-8'))
            before=pending['versions_before']
            now=store.read()['versions']
            role=pending['update_role']
            payload=pending.get('update_trajectory')
            if role is not None and payload is not None and int(now[role])>int(before[role]):
                durable_queue.mark_published(payload['trajectory_id'],role)
    common_env=os.environ.copy(); common_env.update({'PYTHONPATH':str(Path.cwd()),
        'PATH':str(Path(sys.executable).parent)+os.pathsep+common_env.get('PATH',''),
        'VLLM_ALLOW_RUNTIME_LORA_UPDATING':'1','VLLM_USE_FLASHINFER_SAMPLER':'0',
        'VLLM_ATTENTION_BACKEND':'TRITON_ATTN','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
    serve_env=common_env.copy(); serve_env['CUDA_VISIBLE_DEVICES']=str(args.rollout_gpu)
    serve=[str(Path(sys.executable).parent/'vllm'),'serve',args.base,'--served-model-name',args.served_model_name,'--port',str(args.port),'--dtype','bfloat16','--max-model-len','8192','--gpu-memory-utilization','0.35','--enable-lora','--max-lora-rank','16','--return-tokens-as-token-ids','--tensor-parallel-size','1','--attention-backend','TRITON_ATTN']
    processes=[]; files=[]
    try:
        p,f=popen(serve,root/'logs/serve.log',serve_env); processes.append(p); files.append(f)
        import requests
        url=f'http://127.0.0.1:{args.port}'
        deadline=time.time()+args.server_startup_timeout
        while time.time()<deadline:
            try:
                if requests.get(url+'/v1/models',timeout=2).ok: break
            except: pass
            time.sleep(2)
        else: raise RuntimeError('vLLM startup timeout')
        # Learners consume scheduled records and share a lease on each learner GPU.
        current_versions=store.read()['versions']
        role_updates={role:max(0,int(target_updates[role])-int(current_versions[role])) if args.resume
                      else int(target_updates[role]) for role in target_updates}
        role_gpus={'optimizer':args.optimizer_gpu,'planner':args.planner_gpu,'inspector':args.inspector_gpu}
        multi_gpu=all(v is not None for v in role_gpus.values())
        gpu_counts={g:list(role_gpus.values()).count(g) for g in set(role_gpus.values()) if g is not None}
        common_learner=[f'--base',args.base]
        for item in args.learner_arg:
            if '=' not in item: raise SystemExit(f'--learner-arg expects key=value, got {item!r}')
            key,value=item.split('=',1); common_learner += [f'--{key.lstrip("-")}',value]
        for role in ('optimizer','planner','inspector'):
            learner_env=common_env.copy()
            if role_updates[role] <= 0:
                continue
            if args.learner_script:
                learner_script=args.learner_script
                cmd=[py,learner_script,'--role',role,'--queue',queue,'--policy-store',str(root/'policies'),'--workdir',str(root/'learners'),'--max-updates',str(role_updates[role]),'--idle-timeout',str(args.learner_idle_timeout),'--seed',str(args.seed)]+common_learner
            elif args.prefix_value or args.adaptive_prefix_value:
                learner_script='scripts/online_rl/learner_adaptive.py' if args.adaptive_prefix_value else 'scripts/online_rl/learner_prefix.py'
                cmd=[py,learner_script,'--role',role,'--queue',queue,'--policy-store',str(root/'policies'),'--workdir',str(root/'learners'),'--max-updates',str(role_updates[role]),'--idle-timeout',str(args.learner_idle_timeout),'--lr',str(args.learner_lr),'--value-lr',str(args.value_lr),'--seed',str(args.seed),'--actor-epochs',str(args.actor_epochs),'--target-kl',str(args.target_kl),'--min-mask-frac',str(args.min_mask_frac),'--critic-warmup-updates',str(args.critic_warmup_updates)]+common_learner
                if args.adaptive_prefix_value:
                    cmd += ['--beta-kl',str(args.beta_kl)]
            else:
                cmd=[py,'scripts/online_rl/learner.py','--role',role,'--queue',queue,'--policy-store',str(root/'policies'),'--workdir',str(root/'learners'),'--max-updates',str(role_updates[role]),'--idle-timeout',str(args.learner_idle_timeout),'--surrogate',args.surrogate,'--lr',str(args.learner_lr),'--value-lr',str(args.value_lr),'--beta-kl',str(args.beta_kl),'--seed',str(args.seed)]+common_learner
            if multi_gpu:
                learner_env['CUDA_VISIBLE_DEVICES']=str(role_gpus[role])
                cmd += ['--device','cuda:0']
                if gpu_counts.get(role_gpus[role],0)>1:
                    cmd += ['--gpu-lock',str(root/f'learner_gpu_{role_gpus[role]}.lock')]
            else:
                cmd += ['--gpu-lock',str(root/'learner_gpu.lock'),'--device','cuda']
            lp,lf=popen(cmd,root/f'logs/learner_{role}.log',learner_env); processes.append(lp); files.append(lf)
        collector=[py,'scripts/online_rl/collector.py','--queue',queue,'--policy-store',str(root/'policies'),'--base-url',url+'/v1','--episodes',args.episodes,'--pv',str(args.pv),'--seed',str(args.seed),'--K',str(args.K),'--window',str(args.window),'--ablation-mode',args.ablation_mode,'--budget-scale',str(args.budget_scale),'--env-profile',args.env_profile,'--temperature',str(args.temperature),'--inspector-stats',args.inspector_stats]
        if args.collector_out: collector += ['--out',args.collector_out]
        if args.resume: collector += ['--resume']
        if args.schedule_origin is not None: collector += ['--schedule-origin',str(args.schedule_origin)]
        collector += ['--role-schedule',args.role_schedule]
        if args.prefix_value or args.adaptive_prefix_value:
            collector += ['--prefix-value','--portfolio-reward-weight',str(args.portfolio_reward_weight)]
            if args.adaptive_prefix_value: collector += ['--cost-margin-weight','1.5']
        if args.wait_for_role_update: collector += ['--wait-for-target-update']
        cp,cf=popen(collector,root/'logs/collector.log',common_env); processes.append(cp); files.append(cf)
        rc=cp.wait()
        if rc!=0: raise RuntimeError(f'collector rc={rc}')
        learner_processes = processes[1:-1]
        for lp in learner_processes:
            rc=lp.wait(timeout=1800)
            if rc!=0: raise RuntimeError(f'learner rc={rc}')
        manifest=store.read()
        run_manifest={'status':'DONE','episodes':args.episodes,'pv':args.pv,'seed':args.seed,
                      'budget_scale':args.budget_scale,'environment_profile':args.env_profile,
                      'temperature':args.temperature,
                      'beta_kl':None if args.prefix_value and not args.adaptive_prefix_value else args.beta_kl,
                      'inspector_stats_sha256':hashlib.sha256(Path(args.inspector_stats).read_bytes()).hexdigest(),
                      'schedule_origin':args.schedule_origin,'role_schedule':args.role_schedule,
                      'ablation_mode':args.ablation_mode,'resume':args.resume,
                      'target_versions':target_updates,'initial_adapters':{
                          'planner':args.planner_init,'optimizer':args.optimizer_init,
                          'inspector':args.inspector_init},'versions':manifest['versions']}
        (root/'run_manifest.json').write_text(json.dumps(run_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        (root/'DONE').touch()
        print(json.dumps({'status':'DONE','manifest':manifest},ensure_ascii=False,indent=2))
    finally:
        for p in reversed(processes):
            if p.poll() is None: p.terminate()
        for p in reversed(processes):
            try: p.wait(timeout=15)
            except: p.kill()
        for f in files: f.close()

if __name__=='__main__': main()
