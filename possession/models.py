"""Portable JSON models; runtime uses NumPy and never unpickles an object."""
import json
from pathlib import Path
import numpy as np


def canonical(value):
    # +0 and -0 make identical comparisons, but OpenMP may emit either sign in
    # tree thresholds. Normalize this irrelevant sign for byte reproducibility.
    if isinstance(value,dict): return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,list): return [canonical(v) for v in value]
    if isinstance(value,float) and value==0: return 0.0
    return value


def dump(path,value):
    # A network is hundreds of thousands of weights: one per line would be several times larger.
    indent=None if isinstance(value,dict) and value.get('kind')=='tcn' else 2
    Path(path).write_text(json.dumps(canonical(value),indent=indent,allow_nan=False)+'\n')


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))


def export_estimator(estimator,schema,kind,mean=None,scale=None):
    model={'schema_version':1,'kind':kind,**schema}
    if kind=='linear':
        model.update({'mean':mean.tolist(),'scale':scale.tolist(),'coef':estimator.coef_[0].tolist(),'bias':float(estimator.intercept_[0])})
    elif kind=='boosted':
        trees=[]
        for stage in estimator._predictors:
            nodes=stage[0].nodes
            if np.any(nodes['is_categorical']): raise ValueError('Categorical splits not supported')
            trees.append({k:nodes[field].tolist() for k,field in [('feature','feature_idx'),('left','left'),('right','right'),
                         ('threshold','num_threshold'),('value','value'),('leaf','is_leaf')]})
        model.update({'bias':float(estimator._baseline_prediction.ravel()[0]),'trees':trees})
    else: raise ValueError('Unsupported estimator')
    return model


def _convolve(h,weight,bias,dilation=1):
    """1-D convolution along time with zero padding. h: (frames, channels); weight: (out, in, taps)."""
    weight=np.asarray(weight,dtype=np.float64); out=np.zeros((len(h),weight.shape[0]))+np.asarray(bias,dtype=np.float64)
    taps=weight.shape[2]
    for j in range(taps):
        shift=(j-taps//2)*dilation; part=h@weight[:,:,j].T
        if shift==0: out+=part
        elif shift>0:
            if shift<len(h): out[:-shift]+=part[shift:]
        elif -shift<len(h): out[-shift:]+=part[:shift]
    return out


def _sequence_score(x,model):
    """Temporal convolutional network over one uninterrupted sequence of frames; members are averaged."""
    h0=(np.asarray(x,dtype=np.float64)-np.asarray(model['mean']))/np.asarray(model['scale']); total=np.zeros(len(x))
    for member in model['members']:
        h=_convolve(h0,member['inp']['w'],member['inp']['b'])
        for block in member['blocks']:
            inner=np.maximum(_convolve(h,block['w3'],block['b3'],block['dilation']),0)
            h=np.maximum(h+_convolve(inner,block['w1'],block['b1']),0)
        total+=_convolve(h,member['out']['w'],member['out']['b'])[:,0]
    return total/len(model['members'])


def raw_score(x,model):
    if model['kind']=='tcn': return _sequence_score(x,model)
    if model['kind']=='linear':
        return ((x-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['coef'])+model['bias']
    if model['kind']!='boosted': raise ValueError('Unknown model kind')
    out=np.full(len(x),model['bias'],dtype=np.float64)
    for tree in model['trees']:
        leaf=np.asarray(tree['leaf'],bool); feature=np.asarray(tree['feature'],int)
        threshold=np.asarray(tree['threshold']); left=np.asarray(tree['left'],int); right=np.asarray(tree['right'],int)
        indices=np.zeros(len(x),np.int32)
        for _ in range(len(leaf)):
            active=np.flatnonzero(~leaf[indices])
            if not len(active): break
            node=indices[active]
            indices[active]=np.where(x[active,feature[node]]<=threshold[node],left[node],right[node])
        else: raise ValueError('Cyclic model tree')
        out+=np.asarray(tree['value'])[indices]
    return out


def score(x,model):
    if x.shape[1]!=len(model['feature_names']): raise ValueError('Feature width mismatch')
    signs=np.asarray(model['swap_sign'],dtype=np.float32)
    return .5*(raw_score(x,model)-raw_score(x*signs,model))/model.get('temperature',1.)


def sampled_score(x,blocks,model,stride=5):
    out=np.zeros(len(x),np.float64)
    if model['kind']=='tcn':
        # A sequence model reads every frame of a block in order; nothing is subsampled.
        for a,b in blocks: out[int(a):int(b)]=score(x[int(a):int(b)],model)
        return out
    for a,b in blocks:
        idx=np.unique(np.r_[np.arange(a,b,stride),b-1]).astype(int)
        out[a:b]=np.interp(np.arange(a,b),idx,score(x[idx],model))
    return out


def load(path):
    m=json.loads(Path(path).read_text())
    if m.get('schema_version')!=1 or m['kind'] not in ('linear','boosted','tcn'): raise ValueError('Unknown model schema')
    p=len(m['feature_names'])
    if len(set(m['feature_names']))!=p or len(m['swap_sign'])!=p or not np.isin(m['swap_sign'],[-1,1]).all(): raise ValueError('Invalid feature schema')
    if not np.isfinite(m.get('temperature',1)) or m.get('temperature',1)<=0: raise ValueError('Invalid temperature')
    if m['kind']=='linear':
        for name in ('mean','scale','coef'):
            if len(m[name])!=p or not np.isfinite(m[name]).all(): raise ValueError('Invalid linear model')
        if np.any(np.asarray(m['scale'])<=0): raise ValueError('Invalid scale')
    elif m['kind']=='tcn':
        for name in ('mean','scale'):
            if len(m[name])!=p or not np.isfinite(m[name]).all(): raise ValueError('Invalid standardisation')
        if np.any(np.asarray(m['scale'])<=0) or not m['members']: raise ValueError('Invalid sequence model')
        for member in m['members']:
            layers=[(member['inp'],'w','b',p)]; width=np.asarray(member['inp']['w']).shape[0]
            for block in member['blocks']:
                if int(block['dilation'])!=block['dilation'] or block['dilation']<1: raise ValueError('Invalid dilation')
                layers+=[(block,'w3','b3',width),(block,'w1','b1',width)]
            layers.append((member['out'],'w','b',width))
            for layer,w,b,inputs in layers:
                weight=np.asarray(layer[w],dtype=np.float64); bias=np.asarray(layer[b],dtype=np.float64)
                if weight.ndim!=3 or weight.shape[1]!=inputs or weight.shape[2]%2!=1 or bias.shape!=(weight.shape[0],): raise ValueError('Malformed layer')
                if not np.isfinite(weight).all() or not np.isfinite(bias).all(): raise ValueError('Nonfinite layer')
            if np.asarray(member['out']['w']).shape[0]!=1: raise ValueError('Sequence model must end in one output')
    else:
        for tree in m['trees']:
            n=len(tree['leaf'])
            if n<1 or any(len(tree[k])!=n for k in tree): raise ValueError('Malformed tree')
            if not np.isfinite(tree['threshold']).all() or not np.isfinite(tree['value']).all(): raise ValueError('Nonfinite tree')
            for i in range(n):
                if not tree['leaf'][i]:
                    if not (0<=tree['feature'][i]<p and i<tree['left'][i]<n and i<tree['right'][i]<n): raise ValueError('Invalid tree references')
    return m
