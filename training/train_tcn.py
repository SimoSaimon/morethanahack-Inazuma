"""Train the temporal convolutional possession model and choose between it and the best tree model.

Run after training/train.py, which writes the tree candidates and training_report.json to --out.
The network reads the same per-frame features as the trees, but as a sequence: six dilated
convolutions see about 2.5 s on each side of a frame. Several copies trained from different seeds
are averaged. It is exported to JSON and executed with NumPy at run time (possession/models.py);
PyTorch is needed only here.

    python training/train_tcn.py --external-dir work/external --cache work/features --out work/retrained

Never opens the reserved test match or any challenge label.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from possession.data import external_case,code_digest
from possession.models import dump,load,sampled_score,sigmoid
from possession.temporal import decode
from possession.evaluation import assess
from possession.io import sha256

NAME='temporal_cnn'


class Network(nn.Module):
    def __init__(self,inputs,channels,dilations,dropout):
        super().__init__()
        self.inp=nn.Conv1d(inputs,channels,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(channels,channels,3,padding=d,dilation=d),nn.ReLU(),
                                                 nn.Dropout(dropout),nn.Conv1d(channels,channels,1)) for d in dilations])
        self.out=nn.Conv1d(channels,1,1)

    def forward(self,x):
        h=self.inp(x)
        for block in self.blocks: h=torch.relu(h+block(h))
        return self.out(h).squeeze(1)


def numbers(tensor):
    """float32 weights as the shortest decimals that read back to the same float32."""
    return np.vectorize(lambda v:float(f'{v:.9g}'))(tensor.detach().numpy().astype(np.float32)).tolist()


def export_member(net,dilations):
    return {'inp':{'w':numbers(net.inp.weight),'b':numbers(net.inp.bias)},
            'blocks':[{'dilation':int(d),'w3':numbers(b[0].weight),'b3':numbers(b[0].bias),'w1':numbers(b[3].weight),'b1':numbers(b[3].bias)}
                      for d,b in zip(dilations,net.blocks)],
            'out':{'w':numbers(net.out.weight),'b':numbers(net.out.bias)}}


def fit(halves,mean,scale,signs,config,seed):
    """halves: [(standardised features, home label, weight)] per half of each training case."""
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    net=Network(len(mean),config['channels'],config['dilations'],config['dropout'])
    length=config['chunk_frames']; batch=config['batch']
    steps_per_epoch=2*sum(max(1,len(h[0])//length) for h in halves)//batch+1
    opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=config['weight_decay'])
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=config['max_lr'],total_steps=config['epochs']*steps_per_epoch)
    sizes=np.array([len(h[0]) for h in halves],float); chance=sizes/sizes.sum(); losses=[]
    for epoch in range(config['epochs']):
        net.train(); total=0.
        for _ in range(steps_per_epoch):
            xs=[];ys=[];ws=[]
            for _ in range(batch):
                x,y,w=halves[rng.choice(len(halves),p=chance)]; start=int(rng.integers(0,len(x)-length))
                x=x[start:start+length]; y=y[start:start+length]; w=w[start:start+length]
                if rng.random()<.5:                      # the same play with the two teams swapped
                    x=((x*scale+mean)*signs-mean)/scale; y=1-y
                xs.append(x.T);ys.append(y);ws.append(w)
            xb=torch.tensor(np.stack(xs).astype(np.float32)); yb=torch.tensor(np.stack(ys)); wb=torch.tensor(np.stack(ws))
            loss=(nn.functional.binary_cross_entropy_with_logits(net(xb),yb,reduction='none')*wb).sum()/wb.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); schedule.step(); total+=loss.item()
        losses.append(total/steps_per_epoch)
        print(f'seed {seed} epoch {epoch+1}/{config["epochs"]}: loss {losses[-1]:.4f}',flush=True)
    return net.eval(),losses


def validate(model,cases,decoders):
    """Mean official spellF1 with the true game state, at the full 25 Hz, for each decoder."""
    scores={mode:sampled_score(c['x'],c['blocks'],model,stride=model['prediction_stride']) for mode,c in cases.items()}
    trials=[]
    for cost,gap in decoders:
        per_mode={}
        for mode,c in cases.items():
            pred,_=decode(scores[mode],c['quality'],c['state'],c['blocks'],fps=25,switch_cost=cost,gap_penalty=gap)
            per_mode[mode]=assess(c,pred,sigmoid(scores[mode]))
        trials.append({'switch_cost':cost,'gap_penalty':gap,'cases':per_mode,
                       'mean_spellF1':float(np.mean([r['spellF1_with_oracle_state'] for r in per_mode.values()]))})
    return trials


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--external-dir',required=True,type=Path); p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path); args=p.parse_args()
    protocol=json.loads((ROOT/'models/protocol.json').read_text()); config=protocol[NAME]
    report=json.loads((args.out/'training_report.json').read_text())
    if report['feature_code_sha256']!=code_digest() or report['protocol']['train']!=protocol['train']:
        raise ValueError('Run training/train.py first, with the same code and protocol')
    torch.set_num_threads(config['threads']); start=time.perf_counter()
    schema=report['feature_schema']; signs=np.asarray(schema['swap_sign'],np.float32)

    raw=[]; sample=[]
    for match in protocol['train']:
        for mode in protocol['modes']:
            c=external_case(args.external_dir,args.cache,match,mode)
            live=(c['state']==1)&(c['owner']>=0)
            for a,b in c['blocks']: raw.append((c['x'][a:b],(c['owner'][a:b]==0).astype(np.float32),live[a:b].astype(np.float32)))
            sample.append(c['x'][live][::protocol['train_stride']])
            print(f'Prepared {match} {mode}',flush=True)
    sample=np.concatenate(sample); sample=np.r_[sample,sample*signs]      # symmetric, so odd features have mean 0
    mean=sample.mean(axis=0,dtype=np.float64).astype(np.float32); scale=sample.std(axis=0,dtype=np.float64).astype(np.float32); scale[scale<1e-6]=1
    halves=[(((x-mean)/scale).astype(np.float32),y,w) for x,y,w in raw]; del raw,sample

    members=[]; losses={}
    for seed in config['seeds']:
        net,losses[str(seed)]=fit(halves,mean,scale,signs,config,seed)
        members.append(export_member(net,config['dilations']))
    model={'schema_version':1,'kind':'tcn',**schema,'mean':mean.tolist(),'scale':scale.tolist(),'members':members,
           'name':NAME,'temperature':1.0,'prediction_stride':1,'decoder':{'switch_cost':.2,'gap_penalty':2.},
           'training_matches':protocol['train'],'validation_matches':protocol['validation'],'reserved_test_matches':protocol['test'],
           'attribution':'IDSSE; Bassek et al. (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0.'}
    dump(args.out/(NAME+'.json'),model); model=load(args.out/(NAME+'.json'))

    # The exported NumPy model must give what PyTorch gives.
    x=halves[0][0][:3000]
    with torch.no_grad(): reference=net(torch.tensor(x.T[None]))[0].numpy()
    from possession.models import _sequence_score
    single=dict(model,members=[members[-1]])
    parity=float(np.max(np.abs(_sequence_score(x*scale+mean,single)-reference)))
    if parity>1e-3: raise AssertionError(f'Portable model differs from PyTorch by {parity}')

    val={mode:external_case(args.external_dir,args.cache,protocol['validation'][0],mode) for mode in protocol['modes']}
    grid=[(c,g) for c in config['switch_costs'] for g in protocol['gap_penalties']]
    trials=validate(model,val,grid); chosen=max(trials,key=lambda t:t['mean_spellF1'])
    model['decoder']={k:chosen[k] for k in ('switch_cost','gap_penalty')}
    dump(args.out/(NAME+'.json'),model)
    print(f'Validation {NAME}: spellF1 {chosen["mean_spellF1"]:.4f}, cost={chosen["switch_cost"]}, gap={chosen["gap_penalty"]}',flush=True)

    # The best tree model, measured the same way: every frame, 25 Hz decoding.
    tree_name=report['selected_model'] if report['selected_model']!=NAME else report[NAME]['compared_with']
    tree=load(args.out/(tree_name+'.json'))
    tree_trial=validate(tree,val,[(tree['decoder']['switch_cost'],tree['decoder']['gap_penalty'])])[0]
    print(f'Validation {tree_name} at 25 Hz: spellF1 {tree_trial["mean_spellF1"]:.4f}',flush=True)

    winner=NAME if chosen['mean_spellF1']>tree_trial['mean_spellF1'] else tree_name
    (args.out/'possession_model.json').write_bytes((args.out/(winner+'.json')).read_bytes())
    report[NAME]={'config':config,'compared_with':tree_name,'training_loss_per_epoch':losses,'export_max_absolute_logit_error':parity,
                  'validation_25hz':{NAME:chosen,tree_name:tree_trial},'all_decoder_trials':trials,
                  'torch':torch.__version__,'runtime_s':time.perf_counter()-start,
                  'reproducibility':'weights depend on the PyTorch build and thread count; the exported JSON is what the pipeline runs'}
    report['selected_model']=winner; report['selected_model_sha256']=sha256(args.out/'possession_model.json')
    dump(args.out/'training_report.json',report)
    print(json.dumps({'selected_model':winner,'validation_spellF1_25hz':{NAME:chosen['mean_spellF1'],tree_name:tree_trial['mean_spellF1']},
                      'model_sha256':report['selected_model_sha256']},indent=2))


if __name__=='__main__': main()
