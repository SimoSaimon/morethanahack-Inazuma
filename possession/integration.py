"""Strict adapter for the team's in-play / stopped / unknown frame labels."""
import csv
import json
from pathlib import Path
import numpy as np
from .temporal import decode

STATE_NAMES={'in_play':1,'in gioco':1,'stopped':0,'interrotta':0,'unknown':-1,'incerta':-1}


def read_states(path,metadata_path,n_frames,fps=25,timeline='BAR-ATH_tactical'):
    metadata=json.loads(Path(metadata_path).read_text())
    expected={'schema_version':1,'timeline':timeline,'fps':fps,'frame_origin':0,'n_frames':n_frames}
    for key,value in expected.items():
        if metadata.get(key)!=value: raise ValueError(f'State metadata {key}: expected {value!r}')
    states=np.full(n_frames,-1,np.int8)
    with open(path,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['frame','state']: raise ValueError('Required state CSV header: frame,state')
        count=0
        for i,row in enumerate(reader):
            if set(row)!={'frame','state'} or any(v is None for v in row.values()): raise ValueError('Malformed state row')
            if i>=n_frames or row['frame']!=str(i): raise ValueError('State frames must be complete, ordered and zero-based')
            state=row['state'].strip().lower()
            if state not in STATE_NAMES: raise ValueError('Unsupported state '+state)
            states[i]=STATE_NAMES[state]; count+=1
    if count!=n_frames: raise ValueError('Incomplete state stream')
    return states,metadata


def fuse(p_home,quality,states,blocks,decoder,fps=25):
    p_home=np.asarray(p_home,dtype=float); quality=np.asarray(quality,dtype=float)
    if not np.isfinite(p_home).all() or not np.isfinite(quality).all(): raise ValueError('Nonfinite ownership or quality')
    if np.any((p_home<0)|(p_home>1)) or np.any((quality<0)|(quality>1)): raise ValueError('Probabilities/quality must be in [0,1]')
    p=p_home.clip(1e-7,1-1e-7)
    return decode(np.log(p/(1-p)),quality,np.asarray(states),blocks,fps=fps,with_margin=True,**decoder)


def write_submission(path,labels):
    names=np.array(['H','A','D','X'])
    with open(path,'w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n'); writer.writerow(['frame','label'])
        writer.writerows(enumerate(names[labels]))


def read_ownership(path,n_frames):
    p=np.zeros(n_frames); q=np.zeros(n_frames)
    with open(path,newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f); count=0
        if not set(['frame','p_home','p_away','quality']).issubset(reader.fieldnames or []): raise ValueError('Invalid ownership schema')
        for i,row in enumerate(reader):
            if i>=n_frames or row['frame']!=str(i): raise ValueError('Unaligned ownership stream')
            p[i]=float(row['p_home']); q[i]=float(row['quality']); away=float(row['p_away'])
            if not np.isfinite(away) or not 0<=away<=1 or abs(p[i]+away-1)>2e-8: raise ValueError('Ownership probabilities must sum to one')
            count+=1
    if count!=n_frames: raise ValueError('Incomplete ownership stream')
    return p,q
