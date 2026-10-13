"""Shared contracts for single-stream SPO and single-rollout PPO."""
from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from pathlib import Path

def business_channels(delta,cpa_cap=100.0):
    f=float(delta['F']);c=float(delta['C']);n=float(delta['N'])
    d=c-float(delta['M']) if 'M' in delta else c-cpa_cap*n
    return {'F':f,'C':c,'N':n,'D':d}

def business_reward(ch,lambda_d=1.0,illegal=0,forced=0,w_illegal=1.0,w_forced=.05):
    return float(ch['F'])/10000.0-lambda_d*float(ch['D'])/10000.0-w_illegal*illegal-w_forced*forced

def normalize(values,clip=4.0,std_floor=1e-6):
    xs=[float(x) for x in values]
    if not xs:return [],False
    m=sum(xs)/len(xs);s=math.sqrt(sum((x-m)**2 for x in xs)/len(xs))
    if s<std_floor:return [0.0]*len(xs),False
    return [max(-clip,min(clip,(x-m)/s)) for x in xs],True

def tracker_key(transition):
    role=transition.role; tick=int(transition.tick)
    metrics=transition.observation_before.get('metrics',{});acct=metrics.get('per_acct',{}).get(transition.account,{})
    cost=float(acct.get('cost',0));budget=max(float(acct.get('budget',1)),1)
    target_cost=float(acct.get('target_cost',0))
    util=min(int((cost/budget)*4),4)
    margin=min(max(int((cost/max(target_cost,1)-1)*4)+2,0),4)
    stage=min(tick//6,7)
    guidance='g' if 'Inspector' in str(transition.observation_before.get('user','')) else 'n'
    fine=f'{role}|{transition.account}|s{stage}|u{util}|m{margin}|{guidance}'
    coarse=f'{role}|s{stage}|u{util}|{guidance}'
    return fine,coarse,role

@dataclass
class EMAState:
    total:float=0.;weight:float=0.;last_prompt:list|None=None;last_tokens:list|None=None;last_logprobs:list|None=None;count:int=0
    @property
    def value(self):return self.total/self.weight if self.weight else 0.

class EMATracker:
    def __init__(self,path,rho_min=.875,rho_max=.96,d_half=.06):
        self.path=Path(path);self.rho_min=rho_min;self.rho_max=rho_max;self.d_half=d_half;self.states={}
        if self.path.exists():
            raw=json.loads(self.path.read_text());self.states={k:EMAState(**v) for k,v in raw['states'].items()}
    def lookup(self,keys):
        for level,key in enumerate(keys):
            if key in self.states:return self.states[key].value,key,level,False
        return 0.,keys[-1],len(keys)-1,True
    def rho(self,drift):return min(self.rho_max,max(self.rho_min,2**(-float(drift)/self.d_half)))
    def update(self,key,reward,rho,prompt,tokens,logprobs):
        s=self.states.setdefault(key,EMAState());s.total=rho*s.total+reward;s.weight=rho*s.weight+1;s.count+=1;s.last_prompt=list(prompt);s.last_tokens=list(tokens);s.last_logprobs=list(logprobs)
    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps({'schema':'sspo-ema-v1','states':{k:asdict(v) for k,v in self.states.items()}},indent=2));tmp.replace(self.path)
