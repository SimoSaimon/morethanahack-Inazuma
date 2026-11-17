"""Reserved external-match evaluation of the frozen ownership component."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import argparse
import json
from pathlib import Path
import time
import numpy as np
from possession.data import external_case
from possession.models import load,sampled_score,sigmoid
from possession.temporal import decode
from possession.evaluation import assess
from possession.io import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--external-dir',required=True,type=Path); p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--models',default=str(ROOT/'models'),type=Path)
    p.add_argument('--out',required=True,type=Path); args=p.parse_args()
    training=json.loads((args.models/'training_report.json').read_text()); protocol=training['protocol']
    frozen=sha256(args.models/'possession_model.json')
    if frozen!=training['selected_model_sha256']: raise ValueError('Selected model changed after validation')
    start=time.perf_counter(); results={}; timings={}; evidence={}
    variants=['nearest_carried','nearest_temporal','symmetric_logistic','symmetric_boosted_small','symmetric_boosted_medium','symmetric_boosted_large','temporal_cnn']
    for match in protocol['test']:
        for mode in protocol['modes']:
            case=external_case(args.external_dir,args.cache,match,mode)
            key=match+'/'+mode; results[key]={}; timings[key]={}
            evidence[key]={'frames':len(case['owner']),'live_frames':int(case['state'].sum()),
                           'ball_seen_fraction':float(case['ball_seen'].mean()),'source':json.loads(str(case['source']))}
            for name in variants:
                tick=time.perf_counter()
                if name=='nearest_carried':
                    pred=np.full(len(case['owner']),3,np.int8)
                    for a,b in case['blocks']:
                        obs=case['ball_seen'][a:b]&(np.abs(case['nearest_logit'][a:b])>1e-6)
                        indices=np.maximum.accumulate(np.where(obs,np.arange(b-a),0))
                        pred[a:b]=np.where(case['state'][a:b]==0,2,np.where(case['nearest_logit'][a:b][indices]>=0,0,1))
                    prob=None
                else:
                    if name=='nearest_temporal':
                        scores=case['nearest_logit'].astype(float)
                        selected=training['validation'][name]; decoder={k:selected[k] for k in ('switch_cost','gap_penalty')}
                    else:
                        # the selected model is stored once, as possession_model.json
                        model=load(args.models/('possession_model.json' if name==training['selected_model'] else name+'.json'))
                        scores=sampled_score(case['x'],case['blocks'],model,stride=model['prediction_stride']); decoder=model['decoder']
                    pred,_=decode(scores,case['quality'],case['state'],case['blocks'],fps=25,**decoder)
                    prob=sigmoid(scores)
                timings[key][name]=time.perf_counter()-tick
                results[key][name]=assess(case,pred,prob)
                print(f'{key} {name}: spellF1={results[key][name]["spellF1_with_oracle_state"]:.4f}',flush=True)
            # Integration sensitivity: same frozen best model, state errors added in
            # one-second blocks independently of ownership labels. No parameter changes.
            model=load(args.models/'possession_model.json')
            scores=sampled_score(case['x'],case['blocks'],model,stride=5)
            noisy_state=case['state'].copy(); rng=np.random.default_rng(2027)
            for a,b in case['blocks']:
                starts=np.arange(a,b,25); chosen=rng.choice(starts,max(1,len(starts)//50),replace=False)
                for t in chosen:noisy_state[t:min(t+25,b)]=-1
            p,_=decode(scores,case['quality'],noisy_state,case['blocks'],fps=25,**model['decoder'])
            results[key]['selected_with_2pct_unknown_state']=assess(case,p,sigmoid(scores))
    summary={name:{'mean_spellF1':float(np.mean([v[name]['spellF1_with_oracle_state'] for v in results.values()])),
                   'mean_team_accuracy':float(np.mean([v[name]['team_accuracy_on_live'] for v in results.values()]))} for name in variants}
    report={'scope':'H/A component with oracle in_play/stopped labels; NOT end-to-end challenge accuracy',
            'selected_model_frozen_before_test':training['selected_model'],'frozen_model_sha256':frozen,
            'test_matches':protocol['test'],'feature_evidence':evidence,'results':results,'summary':summary,
            'inference_and_decode_s':timings,'runtime_s':time.perf_counter()-start,
            'limitations':['One held-out match; synthetic occlusions are not actual Metrica errors.',
                           'Final selection remains the validation winner regardless of this test.',
                           'Real Smart Tagging and actual game-state predictions are not part of this test.']}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
