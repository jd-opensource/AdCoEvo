"""Single-rollout PPO prefix-credit primitives."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrefixCreditHead(nn.Module):
    def __init__(self,hidden_size,mid=256):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,mid),nn.Tanh(),nn.Linear(mid,1))
    def forward(self,h):return self.net(h.float()).squeeze(-1)

def critic_loss(logits,label,lambda_brier=1.,lambda_prompt=1.):
    y=torch.full_like(logits,float(label));q=torch.sigmoid(logits);per=F.binary_cross_entropy_with_logits(logits,y,reduction='none')+lambda_brier*(q-y).pow(2)
    return per.mean()+lambda_prompt*per[0],q

def token_advantages(q,label,lambda_terminal=1.):
    terminal=lambda_terminal*(float(label)-q[-1]);return (q[1:]-q[:-1]+terminal).detach()

def business_pass(ch,threshold):
    return int(float(ch['F'])>=float(threshold['F_floor']) and float(ch['N'])>=float(threshold['N_floor']) and float(ch['D'])<=0 and int(ch.get('illegal',0))==0 and abs(float(ch.get('budget_error',0)))<=1e-6)
