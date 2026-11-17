"""Possession continuity conditioned on game-state labels."""
import numpy as np
from .numeric import runs

IN_PLAY=1
STOPPED=0
UNKNOWN=-1


def decode(logit_home,quality,states,blocks,fps=25,switch_cost=.8,gap_penalty=2.,with_margin=False):
    if not np.isfinite([fps,switch_cost,gap_penalty]).all() or fps<=0 or switch_cost<0 or gap_penalty<0: raise ValueError('Invalid decoder parameters')
    if not (len(logit_home)==len(quality)==len(states)): raise ValueError('Unaligned arrays')
    if not np.isfinite(logit_home).all() or not np.isfinite(quality).all(): raise ValueError('Nonfinite predictions')
    if not np.isin(states,[-1,0,1]).all(): raise ValueError('Unknown external state')
    last=0
    for a,b in blocks:
        if int(a)!=a or int(b)!=b or not (last<=a<b<=len(states)): raise ValueError('Invalid or overlapping timeline blocks')
        last=b
    labels=np.full(len(states),3,np.int8); margins=np.zeros(len(states))
    for a,b in blocks:
        labels[a:b][states[a:b]==STOPPED]=2
        # An explicit stoppage resets ownership. An unknown frame carries no team
        # evidence, preserves latent continuity, and is exported as X.
        starts,stops=runs(states[a:b]!=STOPPED)
        for s,e in zip(starts+a,stops+a):
            logits=logit_home[s:e]; active=states[s:e]==IN_PLAY
            home=-np.logaddexp(0,-logits)/fps*active
            away=-np.logaddexp(0,logits)/fps*active
            cost=switch_cost*(1+gap_penalty*(1-np.clip(quality[s:e],0,1)))
            n=e-s; forward=np.empty((n,2)); back=np.zeros((n,2),np.int8)
            forward[0]=[home[0],away[0]]
            for t in range(1,n):
                p,q=forward[t-1]; c=cost[t]
                back[t,0]=q-c>p; back[t,1]=q>=p-c
                forward[t]=[max(p,q-c)+home[t],max(q,p-c)+away[t]]
            state=int(forward[-1,1]>forward[-1,0]); path=np.empty(n,np.int8); path[-1]=state
            for t in range(n-2,-1,-1): state=int(back[t+1,state]); path[t]=state
            labels[s:e]=np.where(active,path,3)
            if with_margin:
                u=v=0.
                for t in range(n-1,-1,-1):
                    margins[s+t]=abs(forward[t,0]+u-forward[t,1]-v)
                    c=cost[t]
                    u,v=max(home[t]+u,away[t]+v-c),max(away[t]+v,home[t]+u-c)
    return labels,margins
