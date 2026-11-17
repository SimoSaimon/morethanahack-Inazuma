"""Estimate team possession conditional on active play."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import argparse
import csv
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from possession.io import load_tracking,load_tags,sha256
from possession.cleaning import fix_goalkeeper_team
from possession.features import features
from possession.models import load,sampled_score,sigmoid
from possession.temporal import decode
from possession.data import code_digest
from possession.timeline import BLOCKS,N_FRAMES

OFFENSE={'BUILD UP','PROGRESSION','CREATING CHANCES','ATTACKING TRANSITION'}
DEFENSE={'DEFENDING IN ATTACKING THIRD','DEFENDING IN MIDDLE THIRD','DEFENDING IN DEFENSIVE THIRD','DEFENSIVE TRANSITION'}


def tag_evidence(events,n,team_names):
    values=np.zeros((n,2),np.float32); lookup={name.casefold():i for i,name in enumerate(team_names)}
    for ev in events:
        if ev['code'] not in OFFENSE|DEFENSE: continue
        team=lookup.get((ev['tags'].get('Team') or '').casefold())
        if team is None:continue
        if ev['code'] in DEFENSE: team=1-team
        a,b=ev['start'],ev['stop']; size=b-a
        ramp=np.minimum(np.arange(size)+1,np.arange(size,0,-1))/15
        values[a:b,team]=np.maximum(values[a:b,team],np.minimum(ramp,1))
    return values[:,0]-values[:,1],np.minimum(values[:,0],values[:,1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True,type=Path); p.add_argument('--out',required=True,type=Path)
    p.add_argument('--cache',default='work/cache'); p.add_argument('--model',default=str(ROOT/'models/possession_model.json'))
    p.add_argument('--no-tags',action='store_true'); p.add_argument('--tag-weight',type=float,default=.6)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if not np.isfinite(args.tag_weight) or args.tag_weight<0: p.error('tag weight must be finite and nonnegative')
    start=time.perf_counter(); model=load(args.model); stem=args.data_dir/'tactical/BAR-ATH_tactical'
    training_report=json.loads((Path(args.model).parent/'training_report.json').read_text())
    if training_report['feature_code_sha256']!=code_digest():
        raise ValueError('Feature implementation differs from training; reproduce the model before using it')
    raw,metadata,provenance=load_tracking(Path(str(stem)+'_atd.xml'),Path(str(stem)+'_atd.txt'),args.cache)
    if len(raw)!=N_FRAMES or metadata['fps']!=25: raise ValueError('Expected BAR-ATH tactical at 25 fps')
    x=np.zeros((N_FRAMES,len(model['feature_names'])),np.float32)
    quality=np.zeros(N_FRAMES,np.float32); seen=np.zeros(N_FRAMES,bool); nearest=np.zeros(N_FRAMES,np.float32)
    times={'load_s':time.perf_counter()-start}; tick=time.perf_counter()
    moved=[]
    for a,b in BLOCKS:
        block,fixes=fix_goalkeeper_team(raw[a:b],metadata['team_of_track'])
        moved+=[dict(f,start=f['start']+a,end=f['end']+a) for f in fixes]
        xx,schema,d=features(block,metadata['team_of_track'])
        if schema['feature_names']!=model['feature_names'] or schema['swap_sign']!=model['swap_sign']: raise ValueError('Feature schema mismatch')
        x[a:b]=xx; quality[a:b]=d['quality']; seen[a:b]=d['ball_seen']; nearest[a:b]=d['nearest_logit']
    times['features_s']=time.perf_counter()-tick; tick=time.perf_counter()
    tracking_logit=sampled_score(x,BLOCKS,model,model['prediction_stride'])
    times['model_s']=time.perf_counter()-tick
    tag_path=Path(str(stem)+'_smart_tagging.xml')
    events=[] if args.no_tags else load_tags(tag_path,N_FRAMES,25)
    tags,conflict=tag_evidence(events,N_FRAMES,metadata['team_names'])
    # Engineering soft prior, NOT a learned/calibrated tag likelihood. Separately exposed.
    tag_logit=args.tag_weight*tags*(1-.75*quality)
    combined=tracking_logit+tag_logit; p_home=sigmoid(combined)
    tracking_home=sigmoid(tracking_logit)
    tick=time.perf_counter()
    conditional,margin=decode(combined,quality,np.ones(N_FRAMES,np.int8),BLOCKS,with_margin=True,**model['decoder'])
    names=np.array(['H','A','D','X']); graded=conditional!=3
    # The Viterbi margin depends on transition penalties, so a fixed margin cutoff
    # would be misleading across models. These are diagnostic flags, not abstention.
    uncertain=graded&((np.maximum(p_home,1-p_home)<.7)|(~seen&(np.abs(tags)<.1)))
    with (args.out/'ownership.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['frame','p_home','p_away','quality','ball_seen','conditional_team','uncertain',
                         'tracking_p_home','tag_logit','path_margin'])
        for i in range(N_FRAMES):
            writer.writerow([i,f'{p_home[i]:.9f}',f'{1-p_home[i]:.9f}',f'{quality[i]:.6f}',int(seen[i]),names[conditional[i]],int(uncertain[i]),
                             f'{tracking_home[i]:.9f}',f'{tag_logit[i]:.6f}',f'{margin[i]:.6f}'])
    times['decode_and_export_s']=time.perf_counter()-tick; times['total_s']=time.perf_counter()-start
    memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    report={'task':'ownership conditional on live play; game-state labels supplied by the state component',
            'model_name':model['name'],'model_sha256':sha256(args.model),'ownership_sha256':sha256(args.out/'ownership.csv'),
            'feature_code_sha256':code_digest(),'decoder':model['decoder'],
            'input_provenance':provenance,'rows':N_FRAMES,'timeline':'BAR-ATH_tactical','fps':25,'frame_origin':0,
            'home_team':metadata['team_names'][0],'away_team':metadata['team_names'][1],
            'tags_enabled':not args.no_tags,'tag_weight':args.tag_weight,'tag_weight_origin':'engineering prior, not validated on Metrica tags',
            'tag_source_sha256':None if args.no_tags else sha256(tag_path),'graded_ball_seen_fraction':float(seen[graded].mean()),
            'uncertain_graded_frames':int(uncertain.sum()),'tag_conflict_graded_frames':int(np.sum(conflict[graded]>0)),
            'timings':times,'peak_rss_mib':memory,'numpy':np.__version__,'python':sys.version.split()[0],
            'goalkeeper_frames_moved':moved,'challenge_accuracy':'unknown','state_module_integrated':False}
    (args.out/'run_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
