import hashlib
import numpy as np


BALL_SIGMA=6   # frames; see smooth_ball


def smooth_ball(raw,sigma=BALL_SIGMA):
    """Gaussian-smooth the ball inside each visible run, as the BAR-ATH tracker appears to do.

    Chosen without labels, by matching ball kinematics: velocity changes over 3 frames have
    quantiles (50/75/90/95/99%) 0.8/1.5/2.2/2.7/3.7 m/s in BAR-ATH and 0.6/1.2/3.2/7.0/16.4 in
    IDSSE; IDSSE smoothed with sigma 6 gives 0.7/1.3/2.3/3.0/4.2."""
    out=raw.copy(); ball=out[:,-1,:2]; seen=~np.isnan(ball[:,0])
    r=int(4*sigma); k=np.exp(-.5*(np.arange(-r,r+1)/sigma)**2)
    d=np.diff(np.r_[0,seen.astype(np.int8),0])
    for a,b in zip(np.where(d==1)[0],np.where(d==-1)[0]):
        weight=np.convolve(np.ones(b-a),k,'full')[r:r+b-a]
        for j in (0,1): ball[a:b,j]=np.convolve(ball[a:b,j].astype(np.float64),k,'full')[r:r+b-a]/weight
    return out


def degrade(raw,teams,pitch,match,half,mode,fps=25):
    if mode=='clean': return smooth_ball(raw)
    if mode not in ('occluded','severe'): raise ValueError('Unknown stress mode')
    seed=int(hashlib.sha256(f'ownership-v1:{match}:{half}:{mode}'.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed); x=raw.copy(); n=len(x)
    cursor=0; missing=False
    while cursor<n:
        low,high=((20,60) if missing else (4,24)) if mode=='severe' else ((4,24) if missing else (8,32))
        length=int(rng.integers(low*fps,high*fps+1))
        if missing:x[cursor:cursor+length,-1,:]=np.nan
        cursor+=length; missing=not missing
    for j in range(x.shape[1]-1):
        cursor=int(rng.integers(1,1000))
        while cursor<n:
            length=int(rng.integers(100,501)); x[cursor:cursor+length,j,:]=np.nan
            cursor+=length+int(rng.integers(400,2001))
    x[:,:,:2]+=rng.normal(0,.03 if mode=='occluded' else .08,size=x[:,:,:2].shape).astype(np.float32)/np.asarray(pitch,np.float32)
    # Independent permutations within each team simulate fragmented identities.
    for a in range(0,n,20*fps):
        b=min(n,a+20*fps)
        for team in (0,1):
            indices=np.flatnonzero(np.asarray(teams)==team)
            x[a:b,indices,:]=x[a:b,rng.permutation(indices),:]
    return smooth_ball(x)
