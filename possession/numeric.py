import numpy as np


def runs(mask):
    d=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8))
    return np.flatnonzero(d==1),np.flatnonzero(d==-1)


def rolling_sum(x,half):
    x=np.asarray(x)
    c=np.concatenate([np.zeros((1,)+x.shape[1:],dtype=np.float64),np.cumsum(x,axis=0,dtype=np.float64)])
    ix=np.arange(len(x)); lo=np.maximum(0,ix-half); hi=np.minimum(len(x),ix+half+1)
    return c[hi]-c[lo]


def rolling_stats(x,half):
    x=np.asarray(x,dtype=np.float64)
    ok=np.isfinite(x); count=rolling_sum(ok,half)
    z=np.where(ok,x,0)
    mean=np.divide(rolling_sum(z,half),count,out=np.full_like(z,np.nan),where=count>0)
    variance=np.divide(rolling_sum(z*z,half),count,out=np.full_like(z,np.nan),where=count>0)-mean*mean
    width=rolling_sum(np.ones(len(x)),half)
    if x.ndim>1: width=width[:,None]
    return mean,np.sqrt(np.maximum(variance,0)),count/width


def centered_velocity(xy,valid,fps,lag,jump_limit):
    """No differentiation across even one missing frame or an impossible step."""
    xy=np.asarray(xy)
    velocity=np.full(xy.shape,np.nan,dtype=np.float32)
    if len(xy)<=2*lag: return velocity
    steps=np.linalg.norm(np.diff(xy,axis=0),axis=-1)*fps
    bad=np.zeros(valid.shape,dtype=bool)
    bad[1:]=(~valid[1:])|(~valid[:-1])|(steps>jump_limit)
    bad[0]=~valid[0]
    # A jump marks both sides of a speed window unusable, including identity switches.
    invalid=rolling_sum(bad,lag)>0
    v=(xy[2*lag:]-xy[:-2*lag])*(fps/(2*lag))
    usable=~invalid[lag:-lag]
    velocity[lag:-lag]=np.where(usable[...,None],v,np.nan)
    return velocity


def run_duration(mask,fps):
    result=np.zeros(len(mask),dtype=np.float32)
    for a,b in zip(*runs(mask)): result[a:b]=(b-a)/fps
    return result


def finite_quantiles(values):
    x=np.asarray(values); x=x[np.isfinite(x)]
    if not len(x): return {'n':0}
    return {'n':int(len(x)),**{str(q):float(np.percentile(x,q)) for q in [0,10,50,90,99,100]}}
