"""Train H/A only. Never opens the reserved test match or challenge labels."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from possession.data import external_case,subsample,code_digest
from possession.models import export_estimator,score,sigmoid,dump
from possession.temporal import decode
from possession.evaluation import assess
from possession.io import sha256


def save(path,x): dump(path,x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--external-dir',required=True,type=Path); p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path); args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'models/protocol.json').read_text())
    start=time.perf_counter(); xx=[]; yy=[]; sources={}
    for match in protocol['train']:
        for mode in protocol['modes']:
            case=external_case(args.external_dir,args.cache,match,mode)
            sources[match]=json.loads(str(case['source'])); schema=json.loads(str(case['schema']))
            sample=subsample(case,protocol['train_stride']); keep=(sample['state']==1)&(sample['owner']>=0)
            xx.append(sample['x'][keep]); yy.append((sample['owner'][keep]==0).astype(int))
            print(f'Prepared training {match} {mode}: {int(keep.sum())} live rows',flush=True)
            del case,sample
    x=np.concatenate(xx); y=np.concatenate(yy); del xx,yy
    signs=np.array(schema['swap_sign'],np.float32)
    x=np.r_[x,x*signs]; y=np.r_[y,1-y]
    val={}
    for mode in protocol['modes']:
        case=external_case(args.external_dir,args.cache,protocol['validation'][0],mode)
        sources[protocol['validation'][0]]=json.loads(str(case['source']))
        val[mode]=subsample(case,5); del case
    candidates={}; fit_times={}; parity={}
    with threadpool_limits(limits=2):
        tick=time.perf_counter(); mean=x.mean(axis=0,dtype=np.float64); scale=x.std(axis=0,dtype=np.float64); scale[scale<1e-6]=1
        linear=LogisticRegression(C=1.,max_iter=400,solver='lbfgs',random_state=2027)
        linear.fit((x-mean)/scale,y)
        candidates['symmetric_logistic']=export_estimator(linear,schema,'linear',mean,scale)
        fit_times['symmetric_logistic']=time.perf_counter()-tick
        from possession.models import raw_score
        parity['symmetric_logistic']=float(np.max(np.abs(raw_score(x[:1000],candidates['symmetric_logistic'])-linear.decision_function((x[:1000]-mean)/scale))))
        for name,leaves,iterations,rate in [('symmetric_boosted_small',7,80,.08),('symmetric_boosted_medium',15,160,.08),('symmetric_boosted_large',31,400,.05)]:
            tick=time.perf_counter()
            estimator=HistGradientBoostingClassifier(max_iter=iterations,max_leaf_nodes=leaves,learning_rate=rate,
                        l2_regularization=1.,min_samples_leaf=50,early_stopping=False,random_state=2027)
            estimator.fit(x,y)
            candidates[name]=export_estimator(estimator,schema,'boosted')
            parity[name]=float(np.max(np.abs(raw_score(x[:1000],candidates[name])-estimator.decision_function(x[:1000]))))
            fit_times[name]=time.perf_counter()-tick
            print(f'Fitted {name} in {fit_times[name]:.2f}s; JSON parity {parity[name]:.3g}',flush=True)
    if max(parity.values())>1e-6: raise AssertionError('Portable model differs from training model')
    report={'protocol':protocol,'sources':sources,'training_rows_after_symmetry':len(y),'feature_schema':schema,
            'feature_code_sha256':code_digest(),'fit_times_s':fit_times,'export_max_absolute_logit_error':parity,
            'validation':{},'all_parameter_trials':{}}
    del x,y
    # Controls are independent of model selection. They use the same oracle game state.
    near_results={}
    for mode,c in val.items():
        pred=np.full(len(c['owner']),3,np.int8)
        for a,b in c['blocks']:
            obs=c['ball_seen'][a:b]&(np.abs(c['nearest_logit'][a:b])>1e-6)
            idx=np.maximum.accumulate(np.where(obs,np.arange(b-a),0))
            pred[a:b]=np.where(c['state'][a:b]==0,2,np.where(c['nearest_logit'][a:b][idx]>=0,0,1))
        near_results[mode]=assess(c,pred)
    report['validation']['nearest_carried']={'cases':near_results,'mean_spellF1':float(np.mean([r['spellF1_with_oracle_state'] for r in near_results.values()]))}
    best=None
    all_names=['nearest_temporal']+list(candidates)
    for name in all_names:
        if name=='nearest_temporal': scores={mode:c['nearest_logit'].astype(float) for mode,c in val.items()}; temperature=1.
        else:
            scores={mode:score(c['x'],candidates[name]) for mode,c in val.items()}
            calibration=[]
            for temperature in (.75,1.,1.5,2.):
                losses=[]
                for mode,c in val.items():
                    keep=(c['state']==1)&(c['owner']>=0)
                    losses.append(float(np.mean((sigmoid(scores[mode][keep]/temperature)-(c['owner'][keep]==0))**2)))
                calibration.append((float(np.mean(losses)),temperature))
            _,temperature=min(calibration)
            scores={k:v/temperature for k,v in scores.items()}; candidates[name]['temperature']=temperature
        trials=[]; selected=None
        for cost in protocol['switch_costs']:
            for gap in protocol['gap_penalties']:
                cases={}
                for mode,c in val.items():
                    pred,_=decode(scores[mode],c['quality'],c['state'],c['blocks'],fps=5,switch_cost=cost,gap_penalty=gap)
                    cases[mode]=assess(c,pred,sigmoid(scores[mode]))
                value=float(np.mean([r['spellF1_with_oracle_state'] for r in cases.values()]))
                trial={'switch_cost':cost,'gap_penalty':gap,'temperature':temperature,'mean_spellF1':value,'cases':cases}
                trials.append(trial)
                if selected is None or value>selected['mean_spellF1']: selected=trial
        report['all_parameter_trials'][name]=trials; report['validation'][name]=selected
        print(f'Validation {name}: spellF1 {selected["mean_spellF1"]:.4f}, cost={selected["switch_cost"]}, gap={selected["gap_penalty"]}',flush=True)
        if name in candidates:
            candidates[name].update({'name':name,'decoder':{k:selected[k] for k in ('switch_cost','gap_penalty')},
                                     'prediction_stride':5,'training_matches':protocol['train'],'validation_matches':protocol['validation'],
                                     'reserved_test_matches':protocol['test'],'attribution':'IDSSE; Bassek et al. (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0.'})
            save(args.out/(name+'.json'),candidates[name])
            if best is None or selected['mean_spellF1']>report['validation'][best]['mean_spellF1']: best=name
    report['selected_model']=best; report['runtime_s']=time.perf_counter()-start
    save(args.out/'possession_model.json',candidates[best])
    report['selected_model_sha256']=sha256(args.out/'possession_model.json')
    report['test_not_opened']=True; report['challenge_labels_not_opened']=True
    save(args.out/'training_report.json',report)
    print(json.dumps({'selected_model':best,'validation_spellF1':report['validation'][best]['mean_spellF1'],
                     'runtime_s':report['runtime_s'],'model_sha256':report['selected_model_sha256']},indent=2))


if __name__=='__main__': main()
