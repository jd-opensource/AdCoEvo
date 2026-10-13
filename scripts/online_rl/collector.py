#!/usr/bin/env python3
"""Collect settled AdCoEvo decisions and feedback across continuous AuctionNet periods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import requests

sys.path.insert(0, "."); sys.path.insert(0, "scripts")

import numpy as np
from transformers import AutoTokenizer

import run_ladder as RL
from agents.llm_backend import LLMClient, extract_json
from agents.period_graph import build_period_graph
from train.online_rl.durable_queue import DurableQueue
from train.online_rl.policy_store import PolicyStore
from train.online_rl.schema import ActionTransition, OnlineTrajectory, SCHEMA_VERSION, trajectory_id
from train.online_rl.settled_buffer import (append_record, empty_buffers, select_batch,
                                      settled_role_record)
from train.coevolution.rewards import (channels as reward_channels,
                                       recovery as recovery_channels,
                                       scalar as inspector_reward, validate_stats)
from train.coevolution.ablation import FeedbackRouter, MODES, effective_target
from train.online_rl.reward_attribution import attributed_reward, metric_delta, scalar_reward
from train.online_rl.budget_regime import ROLE_SCHEDULES, episode_record, scheduled_role as regime_scheduled_role


class TraceClient:
    def __init__(self, base_url, model, tokenizer, temperature=0.7, seed=None):
        self.cli = LLMClient(dict(base_url=base_url, model=model, temperature=temperature, seed=seed,
                                  logprobs=True, enable_thinking=False, max_tokens=512, timeout=90))
        self.tok = tokenizer

    def call(self, system, user, schema):
        msgs=[{"role":"system","content":system},{"role":"user","content":user}]
        ids=self.tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,"keys"): ids=ids["input_ids"]
        if ids and isinstance(ids[0],list): ids=ids[0]
        r=self.cli.complete(system,user,guided_json=schema)
        obj=extract_json(r.text) if r.ok else None
        cids=r.completion_token_ids or []; lps=r.token_logprobs or []
        valid=isinstance(obj,dict) and bool(cids) and len(cids)==len(lps)
        return obj if isinstance(obj,dict) else {}, [int(x) for x in ids], [int(x) for x in cids], [float(x) for x in lps], valid


def settle_previous_diagnosis(pending_traj, current_feedback):
    """Attach the next period's business recovery to the preceding diagnosis."""
    if pending_traj is None:
        return None
    baseline = pending_traj.reward_config["business_channels"]
    diagnoses = [t for t in pending_traj.transitions if t.role == "inspector"]
    if len(diagnoses) != 1:
        raise ValueError("pending trajectory must contain one Inspector diagnosis")
    diagnosis = diagnoses[0]
    delta = recovery_channels(current_feedback, baseline)
    diagnosis.reward = inspector_reward(delta)
    diagnosis.terminal = True
    diagnosis.outcome_after["feedback_recovery"] = delta
    return settled_role_record(pending_traj, "inspector")


def collect_episode(ep, seq, args, versions, clients, tok, memory=None):
    # Context holds the live simulator and trace while LangGraph owns phase order.
    W=args.window
    ctx={"transitions":[],"idx":0,"n_illegal":0,"forced_total":0,"set_total":0,
         "set_requests":0,"request_count":0}

    def warmup(state):
        c=state["context"]
        native, _, _, _, _, _ = RL.native_hold_rollout(
            ep, args.pv, args.seed, args.K, budget_scale=args.budget_scale,
            env_profile=args.env_profile)
        e=RL.make_env(ep, args.pv, args.seed, args.K, budget_scale=args.budget_scale,
                      env_profile=args.env_profile)
        order,groups=RL._accts(e)
        base={u:float(e._units[u]["base_budget"]) for u in e._units}
        e.set_our_strategies(RL._make_strategy)
        uids=[u for a in order for u in groups[a]]
        RL.run_warmup(e,uids,W)
        c.update(native=native,e=e,order=order,groups=groups,base=base,uids=uids,
                 warmup_metrics=RL.collect_metrics(e,order,groups),
                 action_records=RL.new_action_records(),
                 cur={u:"1.0" for u in uids},send={u:"HOLD" for u in uids},t=W)
        return {"context":c}

    def planner(state):
        c=state["context"]
        e,order,groups,base=c["e"],c["order"],c["groups"],c["base"]
        U=RL.unit_metrics(e,order,groups,args.K)
        for a in order:
            c["request_count"]+=1
            user=RL.planner_user(ep,a,groups[a],U,W,memory=memory)
            obj,pids,cids,lps,trainable=clients["planner"].call(RL.PLANNER_SYS,user,RL.PLANNER_SCHEMA)
            alpha,action_legal=RL.parse_planner_alpha(
                obj,len(groups[a]),[base[u] for u in groups[a]])
            c["n_illegal"]+=int(not action_legal)
            before=RL.collect_metrics(e,order,groups)
            budgets=RL.apply_budget_proportions(e,groups[a],alpha,base)
            RL.record_planner_action(c["action_records"],a,groups[a],alpha,budgets,action_legal,
                                     obj.get("alpha"),{u:U[u]["cost"] for u in groups[a]})
            c["transitions"].append(ActionTransition("planner",c["idx"],ep,W,a,pids,cids,lps,obj,
                trainable and action_legal,duration=0,
                observation_before={"user":user,"metrics":before},
                outcome_after={"requested_alpha":obj.get("alpha"),"executed_alpha":alpha,
                               "executed_budgets":budgets,"invalid_reply":not action_legal}))
            c["idx"]+=1
        return {"context":c}

    def optimizer_step(state):
        c=state["context"]
        e,order,groups,uids=c["e"],c["order"],c["groups"],c["uids"]
        cur,send,t=c["cur"],c["send"],c["t"]
        U=RL.unit_metrics(e,order,groups,args.K)
        tick_nodes=[]; before_tick=RL.collect_metrics(e,order,groups)
        for a in order:
            c["request_count"]+=1
            user=RL.optimizer_user(ep,t,a,groups[a],U,cur,(memory or {}).get("optimizer_guidance", ""))
            obj,pids,cids,lps,trainable=clients["optimizer"].call(
                RL.OPTIMIZER_SYS,user,RL.OPTIMIZER_SCHEMA)
            decs,action_legal=RL.parse_optimizer_decisions(obj,len(groups[a]))
            c["n_illegal"]+=int(not action_legal)
            RL.record_optimizer_reply(c["action_records"],a,action_legal)
            c["set_requests"]+=sum(decision!="HOLD" for decision in decs)
            for j,u in enumerate(groups[a]):
                send[u]=decs[j]
            tr=ActionTransition("optimizer",c["idx"],ep,t,a,pids,cids,lps,obj,
                                trainable and action_legal,duration=0,
                                observation_before={"user":user,"metrics":before_tick})
            c["idx"]+=1
            c["transitions"].append(tr); tick_nodes.append(tr)
        done=e.step({u:{"tcpa_level":send[u]} for u in uids}).done
        for uid in uids:
            tcpa_state=e._tcpa_state[uid]
            cur[uid]=str(float(tcpa_state["level"]))
            c["forced_total"]+=int(bool(tcpa_state.get("forced_hold_step",False)))
            c["set_total"]+=int(tcpa_state.get("quota_consumed",0))
            RL.record_optimizer_outcome(c["action_records"],e._units[uid]["account_id"],uid,send[uid],tcpa_state)
        after=RL.collect_metrics(e,order,groups); d=metric_delta(after,before_tick)
        r=scalar_reward(d,args.cost_margin_weight)
        for tr in tick_nodes:
            if args.prefix_value:
                account_delta=metric_delta(after["per_acct"][tr.account],before_tick["per_acct"][tr.account])
                attribution=attributed_reward(account_delta,d,args.portfolio_reward_weight,
                                               cost_margin_weight=args.cost_margin_weight)
                tr.reward=attribution["combined_reward"]
                tr.outcome_after={"account_delta":account_delta,"portfolio_delta":d,"reward_attribution":attribution}
            else:
                tr.reward=r/len(order); tr.outcome_after=d
            tr.terminal=done
            # Each account advances one environment tick before its next decision.
            tr.duration=1
        c["t"]=t+1
        return {"context":c,"done":done}

    def finalize_delivery(c):
        """Settle P/O delivery and freeze its feedback before any new diagnosis."""
        e,order,groups=c["e"],c["order"],c["groups"]
        final=RL.collect_metrics(e,order,groups)
        c["final"]=final
        # Planner gets full suffix reward; optimizer already has dense step rewards.
        total_d=metric_delta(final,c["warmup_metrics"])
        total_r=scalar_reward(total_d,args.cost_margin_weight)
        for tr in c["transitions"]:
            if tr.role=="planner":
                if args.prefix_value:
                    account_delta=metric_delta(final["per_acct"][tr.account],
                                                c["warmup_metrics"]["per_acct"][tr.account])
                    attribution=attributed_reward(account_delta,total_d,args.portfolio_reward_weight,
                                                   cost_margin_weight=args.cost_margin_weight)
                    tr.reward=attribution["combined_reward"]
                    tr.outcome_after.update({"account_delta":account_delta,"portfolio_delta":total_d,
                                             "reward_attribution":attribution})
                else:
                    tr.reward=total_r/len(order)
                tr.duration=0
        c["business_channels"]=reward_channels(
            {"F":final["fix"],"C":final["cost"],"N":final["conv"],
             "M":final["target_cost"]},args.inspector_stats,
            forced=c["forced_total"],set_requests=c["set_requests"])

    def inspector(state):
        """Generate the next-period guidance and register an unsettled diagnosis."""
        c=state["context"]
        e,order,groups,final=c["e"],c["order"],c["groups"],c["final"]
        snap=RL.account_snapshot(e,order,groups)
        user=RL.inspector_user(ep,snap,order,final,memory=memory,
                               action_records=c["action_records"])
        c["request_count"]+=1
        obj,pids,cids,lps,trainable=clients["inspector"].call(RL.INSPECTOR_SYS,user,RL.INSPECTOR_SCHEMA)
        action_legal=RL.valid_inspector_reply(obj)
        c["n_illegal"]+=int(not action_legal)
        c["transitions"].append(ActionTransition("inspector",c["idx"],ep,48,"portfolio",pids,cids,lps,obj,
            trainable and action_legal,
            reward=0.0,duration=0,terminal=False,observation_before={"user":user,"metrics":final},
            outcome_after={"correction":obj,"memory_consumed":memory,
                           "action_records":c["action_records"],"invalid_reply":not action_legal}))
        return {"context":c}

    def end_period(state):
        finalize_delivery(state["context"])
        return inspector(state)

    graph=build_period_graph(warmup,planner,optimizer_step,end_period)
    ctx=graph.invoke({"context":ctx,"done":False},config={"recursion_limit":128})["context"]
    final,native=ctx["final"],ctx["native"]
    tid=trajectory_id(ep,args.collector_id,seq,versions)
    return OnlineTrajectory(tid,SCHEMA_VERSION,max(versions.values()),dict(versions),ep,args.seed,args.pv,args.K,
                         ctx["transitions"],{"F":final["fix"],"C":final["cost"],"N":final["conv"],
                                      "M":final["target_cost"],"cpa":final["cpa"]},
                         {"F":native["fix"],"C":native["cost"],"N":native["conv"],"cpa":native["cpa"]},
                         {"inspector_memory_in":memory,
                          "cost_margin_weight":float(args.cost_margin_weight),
                          "budget_scale":float(args.budget_scale),"illegal":int(ctx["n_illegal"]),
                          "forced":int(ctx["forced_total"]),"set_total":int(ctx["set_total"]),
                          "agent_requests":int(ctx["request_count"]),"set_requests":int(ctx["set_requests"]),
                          "business_channels":ctx["business_channels"]},
                         args.collector_id,time.time())


def scheduled_role(ep,args):
    return regime_scheduled_role(ep, args.schedule_origin, ROLE_SCHEDULES[args.role_schedule])


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as handle:
            json.dump(payload,handle,ensure_ascii=False,indent=2,allow_nan=True)
            handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--queue',required=True); ap.add_argument('--policy-store',required=True)
    ap.add_argument('--base-url',default='http://localhost:8001/v1'); ap.add_argument('--collector-id',default='collector0')
    ap.add_argument('--episodes',default='0-2'); ap.add_argument('--pv',type=int,default=500000); ap.add_argument('--seed',type=int,default=20270001); ap.add_argument('--K',type=int,default=8)
    ap.add_argument('--window',type=int,default=12,help='warm-up ticks W: first W ticks run native budget + all HOLD, agent decisions start at tick W')
    ap.add_argument('--budget-scale',type=float,default=1.0)
    ap.add_argument('--env-profile',choices=['general','sparse','extra_sparse_feedback_stress'],default='sparse')
    ap.add_argument('--out',default=None)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--model-prefix',default='adcoevo')
    ap.add_argument('--temperature',type=float,default=0.7)
    ap.add_argument('--wait-for-target-update',action='store_true')
    ap.add_argument('--update-timeout',type=float,default=1800)
    ap.add_argument('--ablation-mode',choices=MODES,default='full')
    ap.add_argument('--inspector-stats',required=True,
                    help='JSON with frozen training-set mu_F, sigma_F, mu_N and sigma_N')
    ap.add_argument('--prefix-value',action='store_true')
    ap.add_argument('--portfolio-reward-weight',type=float,default=0.2)
    ap.add_argument('--cost-margin-weight',type=float,default=1.0,
                    help='eta in Q=(F-eta*(C-M))/10000, where M uses fixed initial tCPA')
    ap.add_argument('--schedule-origin',type=int,default=None)
    ap.add_argument('--role-schedule',choices=list(ROLE_SCHEDULES),default='round_robin')
    args=ap.parse_args(); eps=[]
    stats_path=Path(args.inspector_stats)
    if not stats_path.is_file():
        raise SystemExit(f'Inspector reward stats file not found: {stats_path}')
    stats_bytes=stats_path.read_bytes()
    args.inspector_stats=json.loads(stats_bytes)
    validate_stats(args.inspector_stats)
    stats_hash=hashlib.sha256(stats_bytes).hexdigest()
    if args.budget_scale<=0:raise SystemExit('--budget-scale must be positive')
    if '-' in args.episodes:
        a,b=args.episodes.split('-'); eps=list(range(int(a),int(b)+1))
    else: eps=[int(x) for x in args.episodes.split(',')]
    if not eps:
        raise SystemExit('--episodes must not be empty')
    if args.schedule_origin is None:
        args.schedule_origin=eps[0]
    trainable_roles={effective_target(scheduled_role(episode,args),args.ablation_mode)
                     for episode in eps}
    trainable_roles.discard(None)
    out=Path(args.out) if args.out else None
    arms_path=state_path=None
    protocol={"episodes":eps,"pv":args.pv,"seed":args.seed,"K":args.K,
              "budget_scale":float(args.budget_scale),"environment_profile":args.env_profile,
              "schedule_origin":args.schedule_origin,"role_schedule":args.role_schedule,
              "ablation_mode":args.ablation_mode,"temperature":float(args.temperature),
              "cost_margin_weight":float(args.cost_margin_weight)}
    protocol['inspector_stats_sha256']=stats_hash
    completed=[];saved_state=None;pending_journal=None;journal_path=None
    if out:
        out.mkdir(parents=True,exist_ok=True);arms_path=out/'arms.jsonl';state_path=out/'resume_state.json'
        journal_path=out/'pending_episode.json'
        if args.resume:
            if not arms_path.exists() or not state_path.exists():
                raise SystemExit('cannot resume: arms.jsonl and resume_state.json are required')
            existing=[json.loads(line) for line in arms_path.open(encoding='utf-8') if line.strip()]
            completed=[int(row['episode']) for row in existing]
            if completed != eps[:len(completed)]:
                raise SystemExit('cannot resume: completed episodes are not a contiguous prefix')
            saved_state=json.loads(state_path.read_text(encoding='utf-8'))
            committed=saved_state.get('completed_episodes',[])
            if saved_state.get('protocol') != protocol or committed != eps[:len(committed)]:
                raise SystemExit('cannot resume: protocol or committed sequence mismatch')
            if journal_path.exists():
                pending_journal=json.loads(journal_path.read_text(encoding='utf-8'))
                if (committed and pending_journal.get('episode') == committed[-1] and
                    completed == committed):
                    journal_path.unlink()
                    pending_journal=None
                else:
                    if (pending_journal.get('protocol') != protocol or
                        len(committed) >= len(eps) or
                        pending_journal.get('episode') != eps[len(committed)] or
                        pending_journal.get('versions_before') != saved_state.get('policy_versions')):
                        raise SystemExit('cannot resume: pending episode journal mismatch')
                    if completed not in (committed, committed+[pending_journal['episode']]):
                        raise SystemExit('cannot resume: result rows do not match pending episode')
            elif completed != committed:
                raise SystemExit('cannot resume: results and committed state disagree')
        elif arms_path.exists() or state_path.exists() or journal_path.exists():
            raise SystemExit(f'output already contains state; use --resume or a fresh directory: {out}')
    elif args.resume:
        raise SystemExit('--resume requires --out')
    from train.paths import tokenizer as _tokenizer_path
    tok=AutoTokenizer.from_pretrained(str(_tokenizer_path()))
    store=PolicyStore(args.policy_store); q=DurableQueue(args.queue)
    loaded={}; pending_traj=None
    role_buffers=empty_buffers()
    feedback=FeedbackRouter(args.ablation_mode,args.seed)
    feedback_journal=Path(args.policy_store).parent/'feedback_mapping.json'
    if out and not args.resume:
        arms_path.touch()
        atomic_json(state_path,{"protocol":protocol,"completed_episodes":[],
                                "feedback_history":[],"feedback_mapping":[],
                                "policy_versions":dict(store.read()['versions']),
                                "pending_trajectory":None,
                                "role_buffers":role_buffers})
    if saved_state:
        if not pending_journal and saved_state.get('policy_versions') != store.read().get('versions'):
            raise SystemExit('cannot resume: PolicyStore versions do not match resume state')
        feedback.history=[(int(ep),mem) for ep,mem in saved_state['feedback_history']]
        feedback.mapping=list(saved_state['feedback_mapping'])
        role_buffers=saved_state['role_buffers']
        pending_payload=saved_state['pending_trajectory']
        if pending_payload is None and saved_state['completed_episodes']:
            raise SystemExit('cannot resume: pending trajectory is missing')
        if pending_payload is not None:
            pending_traj=OnlineTrajectory.from_dict(pending_payload)
        q.reset_incomplete_claims()
    start_index=len(saved_state['completed_episodes']) if saved_state else 0
    for seq,ep in enumerate(eps):
        if seq < start_index:
            continue
        if pending_journal is not None:
            journal=pending_journal
            versions=journal['versions_before']
            traj=OnlineTrajectory.from_dict(journal['trajectory'])
            produced_memory=journal['produced_memory']
            feedback_source_episode=journal['feedback_source_episode']
            feedback.history=[(int(x),m) for x,m in journal['feedback_history']]
            feedback.mapping=list(journal['feedback_mapping'])
            current_feedback=traj.reward_config['business_channels']
            role_buffers=journal['role_buffers_after']
            update_payload=journal.get('update_trajectory')
            dropped_records=journal.get('dropped_buffer_records',[])
            scheduled=journal['scheduled_role'];update_role=journal['update_role']
        else:
            memory, feedback_source_episode=feedback.consume(ep)
            manifest=store.read(); versions=dict(manifest['versions'])
            # Runtime LoRA swap: collector always rolls out the latest published role versions.
            for role,version in versions.items():
                name=f"{args.model_prefix}_{role}"
                if loaded.get(role)==version:
                    continue
                if role in loaded:
                    requests.post(args.base_url.rsplit('/v1',1)[0]+"/v1/unload_lora_adapter",
                                  json={"lora_name":name},timeout=60).raise_for_status()
                requests.post(args.base_url.rsplit('/v1',1)[0]+"/v1/load_lora_adapter",
                              json={"lora_name":name,"lora_path":str(store.adapter(role,version).resolve())},timeout=60).raise_for_status()
                loaded[role]=version
            # Deterministic but distinct sampling stream per training seed/episode/role.
            clients={r:TraceClient(args.base_url,f"{args.model_prefix}_{r}",tok,args.temperature,
                                   seed=args.seed*1000+ep*10+ri)
                     for ri,r in enumerate(('planner','optimizer','inspector'))}
            traj=collect_episode(ep,seq,args,versions,clients,tok,memory=memory); traj.validate()
            insp=[t for t in traj.transitions if t.role=="inspector"][-1]
            obj=insp.parsed_action
            produced_memory=RL.inspector_memory(obj,ep)
            feedback.produce(ep,produced_memory)
            current_feedback=traj.reward_config['business_channels']
            scheduled=scheduled_role(ep,args);update_role=effective_target(scheduled,args.ablation_mode)
            # Every completed period settles Planner and Optimizer nodes. The
            # preceding Inspector node settles only after this period finishes.
            for role in ("planner","optimizer"):
                if role in trainable_roles:
                    append_record(role_buffers,role,settled_role_record(traj,role))
            settled_diagnosis=settle_previous_diagnosis(pending_traj,current_feedback)
            if settled_diagnosis is not None and "inspector" in trainable_roles:
                append_record(role_buffers,"inspector",settled_diagnosis)
            update_payload=None;dropped_records=[]
            if update_role is not None:
                update_payload,dropped_records=select_batch(
                    role_buffers,update_role,int(versions[update_role]),ep)
            traj.reward_config["ablation"]={"mode":args.ablation_mode,
                                               "feedback_source_episode":feedback_source_episode}
            if out:
                atomic_json(journal_path,{"protocol":protocol,"episode":ep,"versions_before":versions,
                                          "trajectory":traj.to_dict(),
                                          "update_trajectory":update_payload,
                                          "role_buffers_after":role_buffers,
                                          "dropped_buffer_records":dropped_records,
                                          "produced_memory":produced_memory,
                                          "feedback_source_episode":feedback_source_episode,
                                          "feedback_history":[[int(x),m] for x,m in feedback.history],
                                          "feedback_mapping":feedback.mapping,
                                          "scheduled_role":scheduled,"update_role":update_role})
        feedback_journal.write_text(json.dumps(feedback.mapping,ensure_ascii=False,indent=2),encoding='utf-8')
        if dropped_records:
            print(f"[collector] ep={ep} discarded settled records: {dropped_records}",flush=True)
        if update_payload is not None:
            q.put(update_payload['trajectory_id'],update_role,update_payload)
        pending_traj=traj
        print(f"[collector] {traj.trajectory_id} ep={ep} target={scheduled} update={update_role} versions={versions} queue={q.stats()}",flush=True)
        if args.wait_for_target_update and update_payload is not None:
            expected=int(versions[update_role])+1; deadline=time.time()+args.update_timeout
            while int(store.read()['versions'][update_role])<expected:
                if time.time()>deadline: raise TimeoutError(f'waiting for {update_role} version {expected}')
                time.sleep(1.0)
            print(f"[collector] ep={ep} observed {update_role} publish v{expected}",flush=True)
        versions_after=dict(store.read()['versions'])
        if out:
            illegal=int(traj.reward_config.get('illegal',0));forced=int(traj.reward_config.get('forced',0))
            set_total=int(traj.reward_config.get('set_total',0))
            rec=episode_record(traj,scheduled,update_role,versions,versions_after,
                               feedback_source_episode,args.budget_scale,illegal,forced,set_total,
                               agent_requests=int(traj.reward_config['agent_requests']),
                               set_requests=int(traj.reward_config['set_requests']),
                               cpa_tolerance=RL.CPA_TOL)
            if ep not in completed:
                with arms_path.open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            atomic_json(out/f'memory_after_{ep}.json',produced_memory)
            atomic_json(state_path,{"protocol":protocol,"completed_episodes":eps[:seq+1],
                                    "feedback_history":[[int(x),m] for x,m in feedback.history],
                                    "feedback_mapping":feedback.mapping,"policy_versions":versions_after,
                                    "pending_trajectory":traj.to_dict(),
                                    "role_buffers":role_buffers})
            journal_path.unlink(missing_ok=True)
            pending_journal=None

if __name__=='__main__': main()
