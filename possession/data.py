import hashlib
import json
from pathlib import Path
import numpy as np
from .idsse import prepare
from .features import features
from .augmentation import degrade
from .io import sha256


def code_digest():
    h=hashlib.sha256()
    for name in ('features.py','flight.py','augmentation.py','numeric.py','idsse.py','data.py'):
        h.update((Path(__file__).parent/name).read_bytes())
    return h.hexdigest()


def external_case(root,cache,match,mode):
    root=Path(root); cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    source=prepare(root/f'{match}_positions.xml',root/f'DFL-MAT-{match}_meta.xml',cache/'raw')
    key=hashlib.sha256((source.name+mode+code_digest()).encode()).hexdigest()
    dest=cache/(key+'.npz')
    if dest.exists():
        with np.load(dest,allow_pickle=False) as f: return {k:f[k] for k in f.files}
    with np.load(source,allow_pickle=False) as d:
        info=json.loads(str(d['metadata'])); xs=[]; owners=[]; states=[]; qs=[]; seen=[]; near=[]; blocks=[]; offset=0
        for half in (0,1):
            raw=degrade(d[f'raw{half}'],d[f'team{half}'],info['pitch_m'],match,half,mode)
            x,schema,diagnostics=features(raw,d[f'team{half}'],info['pitch_m'])
            xs.append(x); owners.append(d[f'owner{half}']); states.append(d[f'state{half}'])
            qs.append(diagnostics['quality']); seen.append(diagnostics['ball_seen']); near.append(diagnostics['nearest_logit'])
            blocks.append([offset,offset+len(x)]); offset+=len(x)
    out={'x':np.concatenate(xs),'owner':np.concatenate(owners),'state':np.concatenate(states),
         'quality':np.concatenate(qs),'ball_seen':np.concatenate(seen),'nearest_logit':np.concatenate(near),
         'blocks':np.array(blocks,dtype=int),'schema':np.array(json.dumps(schema)),'source':np.array(json.dumps(info))}
    np.savez(dest,**out)
    return out


def subsample(case,stride):
    indices=[]; blocks=[]; offset=0
    for a,b in case['blocks']:
        ix=np.arange(a,b,stride); indices.append(ix); blocks.append([offset,offset+len(ix)]); offset+=len(ix)
    ix=np.concatenate(indices)
    out={k:v[ix] if k in ('x','owner','state','quality','ball_seen','nearest_logit') else v for k,v in case.items()}
    out['blocks']=np.array(blocks)
    return out
