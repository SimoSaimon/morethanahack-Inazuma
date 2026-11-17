"""Identity-free, team-equivariant possession features. No game-status labels."""
import numpy as np
from .numeric import centered_velocity,rolling_stats
from .flight import flight_features


def _average(x,valid):
    count=valid.sum(axis=1)
    return np.divide(np.where(valid,x,0).sum(axis=1),count,out=np.zeros(len(x)),where=count>0)


def features(raw,teams,pitch=(105.3,68.),fps=25,chunk=4096):
    n=len(raw); teams=np.asarray(teams); scale=np.asarray(pitch,np.float32)
    if raw.ndim!=3 or raw.shape[1]!=len(teams)+1 or set(teams)!={0,1}: raise ValueError('Invalid tracking/team schema')
    ball=raw[:,-1,:2]*scale
    valid=np.isfinite(ball).all(axis=1)
    valid&=(np.abs(raw[:,-1,0]-.5)<=.75)&(np.abs(raw[:,-1,1]-.5)<=.75)
    steps=np.linalg.norm(np.diff(ball,axis=0),axis=1)*fps
    if n>2:
        isolated=(steps[:-1]>60)&(steps[1:]>60)&(np.linalg.norm(ball[2:]-ball[:-2],axis=1)*fps/2<30)
        valid[1:-1]&=~isolated
    ball=np.where(valid[:,None],ball,np.nan)
    bv=centered_velocity(ball,valid,fps,5,60)
    ball_speed=np.nan_to_num(np.linalg.norm(bv,axis=1),nan=0).clip(0,40)
    keys=['count','spread_x','spread_y','centroid_speed','mean_speed','running','distance_centroid',
          'nearest','second','density','relative_speed','closing','intercept04','intercept08','alignment']
    values=np.zeros((n,2,len(keys)),np.float32)
    nearest=np.full((n,2),np.inf,np.float32)
    for a in range(0,n,chunk):
        b=min(n,a+chunk); l=max(0,a-6); r=min(n,b+6)
        xy=raw[l:r,:-1,:2]*scale; present=np.isfinite(xy).all(axis=2)
        pv=centered_velocity(xy,present,fps,5,13)
        sl=slice(a-l,b-l); xy=xy[sl]; present=present[sl]; pv=pv[sl]
        for team in (0,1):
            idx=np.flatnonzero(teams==team); p=xy[:,idx]; v=pv[:,idx]; ok=present[:,idx]
            count=ok.sum(axis=1); safe=np.maximum(count,1)
            centroid=np.where(ok[:,:,None],p,0).sum(axis=1)/safe[:,None]
            centered=np.where(ok[:,:,None],p-centroid[:,None,:],0)
            spread=np.sqrt((centered**2).sum(axis=1)/safe[:,None])
            speed=np.linalg.norm(v,axis=2); vok=np.isfinite(speed)
            velocity_mean=np.column_stack([_average(v[:,:,k],vok) for k in (0,1)])
            dist=np.linalg.norm(p-ball[a:b,None,:],axis=2)
            dist=np.where(ok&valid[a:b,None],dist,np.inf)
            pick=np.argmin(dist,axis=1); d1=dist[np.arange(b-a),pick]; nearest[a:b,team]=d1
            if len(idx)>1: d2=np.partition(dist,1,axis=1)[:,1]
            else: d2=np.full(b-a,np.inf)
            closest_v=v[np.arange(b-a),pick]; delta=ball[a:b]-p[np.arange(b-a),pick]
            rel=bv[a:b]-closest_v
            relative=np.linalg.norm(rel,axis=1)
            closing=-np.sum(rel*delta,axis=1)/np.maximum(d1,.1)
            alignment=np.sum(bv[a:b]*closest_v,axis=1)/(np.maximum(np.linalg.norm(closest_v,axis=1)*ball_speed[a:b],.1))
            intercept=[]
            for horizon in (.4,.8):
                future_dist=np.linalg.norm(p+np.nan_to_num(v)*horizon-(ball[a:b]+bv[a:b]*horizon)[:,None,:],axis=2)
                future_dist=np.where(ok&np.isfinite(future_dist),future_dist,np.inf)
                intercept.append(future_dist.min(axis=1))
            data=[count/11,spread[:,0]/30,spread[:,1]/20,np.linalg.norm(velocity_mean,axis=1)/4,
                  _average(speed,vok)/4,_average(speed>2,vok),np.linalg.norm(centroid-ball[a:b],axis=1)/50,
                  np.minimum(d1,25)/10,np.minimum(d2,25)/10,np.exp(-dist/3).sum(axis=1),
                  relative/15,closing/15,np.minimum(intercept[0],25)/10,np.minimum(intercept[1],25)/10,alignment]
            values[a:b,team]=np.nan_to_num(np.column_stack(data),nan=0,posinf=0,neginf=0).clip(-5,5)
    # Difference features change sign under H/A swap; context features do not.
    pair=(values[:,0,0]>0)&(values[:,1,0]>0)
    diff=(values[:,0]-values[:,1])*pair[:,None]
    sums=(values[:,0]+values[:,1])/2
    obs_quality=valid*np.minimum(np.minimum(values[:,0,0],values[:,1,0]),1)
    mind=nearest.min(axis=1)
    contact_quality=obs_quality*np.exp(-np.minimum(mind,50)/8)
    geom=np.zeros(n,np.float32); both=np.isfinite(nearest).all(axis=1)
    geom[both]=np.clip((nearest[both,1]-nearest[both,0])/2.5,-3,3)
    entries={f'diff_{k}':diff[:,j] for j,k in enumerate(keys)}
    odd=[True]*len(entries)
    for j,k in enumerate(keys): entries[f'mean_{k}']=sums[:,j]; odd.append(False)
    for k,x in [('ball_seen',valid),('ball_speed',np.log1p(ball_speed)),('observation_quality',obs_quality),
                ('contact_quality',contact_quality)]: entries[k]=x; odd.append(False)
    entries['geometry']=geom; odd.append(True)
    for seconds in (1,3):
        for key in ['diff_nearest','diff_relative_speed','diff_density','diff_running','diff_spread_y','geometry']:
            entries[f'{key}_window{seconds}']=rolling_stats(entries[key],int(seconds*fps/2))[0]; odd.append(True)
    # Memory of observations, not interpolation of coordinates or target labels.
    anchor=valid&pair&(contact_quality>.1); ix=np.arange(n)
    previous=np.maximum.accumulate(np.where(anchor,ix,-1))
    following=np.minimum.accumulate(np.where(anchor,ix,n)[::-1])[::-1]
    for side,idx,exists,age in [('past',previous,previous>=0,(ix-previous)/fps),
                               ('future',following,following<n,(following-ix)/fps)]:
        safe=idx.clip(0,max(0,n-1)); strength=np.where(exists,contact_quality[safe],0)
        for tau in (3,12):
            entries[f'memory_{side}_{tau}s']=geom[safe]*strength*np.exp(-age/tau); odd.append(True)
        entries[f'age_{side}']=np.where(exists,np.minimum(age,60)/60,1); odd.append(False)
    cleaned=np.array(raw,dtype=np.float32,copy=True); cleaned[~valid,-1,:]=np.nan
    extra,extra_names,extra_sign=flight_features(cleaned,teams,pitch,fps)
    for j,k in enumerate(extra_names): entries[k]=extra[:,j]; odd.append(extra_sign[j]<0)
    x=np.column_stack(list(entries.values())).astype(np.float32)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite possession features')
    meta={'feature_names':list(entries),'swap_sign':[-1 if v else 1 for v in odd]}
    diagnostics={'ball_seen':valid,'quality':obs_quality.astype(np.float32),'contact_quality':contact_quality.astype(np.float32),
                 'nearest_logit':geom.astype(np.float32),'players_home':np.rint(values[:,0,0]*11).astype(int),
                 'players_away':np.rint(values[:,1,0]*11).astype(int)}
    return x,meta,diagnostics
