"""Check the game-state component against the official DFL BallStatus of the IDSSE matches.

The same five matches as the ownership model (training/download_external.py). Two ball conditions:
`clean` (DFL ball, almost always seen) and `occluded` (augmentation.degrade: the ball removed in
4-24 s blocks, ~40% of frames, close to BAR-ATH's 41%). Player speeds always come from the clean
positions, as BAR-ATH's tracker supplies them directly. No Smart Tagging exists for IDSSE.

Headline: possession spellF1 with an oracle possession (true H/A where we say the ball is alive,
D where we say it is dead). 1.0 would mean the dead-ball component costs nothing.

    python evaluation/evaluate_dead_ball.py --external-dir work/external --cache work/features
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from possession import dead_ball as db
from possession.idsse import prepare
from possession.augmentation import degrade
sys.path.insert(0,str(ROOT/'third_party/metrica/evaluation'))
import possession_metrics as pm  # noqa: E402

MATCHES=('J03WOH','J03WOY','J03WMX','J03WPY','J03WQQ')
MODES=('clean','occluded')
FPS=db.FPS


def cases(external_dir,cache):
    """(match, half, mode, pitch, ball, players, state, owner) for every half and ball condition."""
    for match in MATCHES:
        z=np.load(prepare(Path(external_dir)/f'{match}_positions.xml',Path(external_dir)/f'DFL-MAT-{match}_meta.xml',Path(cache)/'raw'))
        pitch=tuple(json.loads(str(z['metadata']))['pitch_m'])
        for half in (0,1):
            raw=z[f'raw{half}']; team=z[f'team{half}']
            xy=raw[:,:-1,:2]*np.asarray(pitch); speed=np.full(xy.shape[:2],np.nan)
            speed[2:-2]=np.linalg.norm(xy[4:]-xy[:-4],axis=-1)*FPS/4
            players=np.concatenate([raw[:,:-1,:2],speed[...,None]],axis=2).astype(np.float64)
            for mode in MODES:
                ball=degrade(raw,team,pitch,'DFL-MAT-'+match,half,mode)[:,-1,:].astype(np.float64)
                yield match,half,mode,pitch,ball,players,z[f'state{half}'],z[f'owner{half}']


def labels(dead,owner):
    idx=np.where(owner>=0,np.arange(len(owner)),0); np.maximum.accumulate(idx,out=idx)
    return pm.label_codes(np.where(dead,'D',np.where(owner[idx]==0,'H','A')))


def score(dead,state,owner):
    gt_dead=state==0
    spell=pm.spell_scores(labels(gt_dead,owner),labels(dead,owner),[(0,len(state))])['mean_f1']
    g=[(a,b,0) for a,b in db.runs(gt_dead)]; q=[(a,b,0) for a,b in db.runs(dead)]
    ov=pm.spell_overlaps(g,q); f1=[]
    for th in pm.SPELL_IOUS:
        tp,fp,fn=pm.match_spells(ov,len(g),len(q),th); f1.append(2*tp/(2*tp+fp+fn) if tp else 0.)
    # boundary errors: each true stoppage against the predicted one overlapping it most
    ds,de=[],[]
    for a,b,_ in g:
        best=max(((min(b,y)-max(a,x),x,y) for x,y,_ in q if x<b and y>a),default=None)
        if best: ds.append(best[1]-a); de.append(best[2]-b)
    return {'spell':spell,'dead_spell':float(np.mean(f1)),'n_gt':len(g),'n_pred':len(q),'d_start':ds,'d_end':de}


def summary(rows):
    ds=np.concatenate([r['d_start'] for r in rows]); de=np.concatenate([r['d_end'] for r in rows])
    return {'spellF1':float(np.mean([r['spell'] for r in rows])),'deadF1':float(np.mean([r['dead_spell'] for r in rows])),
            'stoppages_true':sum(r['n_gt'] for r in rows),'stoppages_pred':sum(r['n_pred'] for r in rows),
            'start_median':float(np.median(ds)),'start_within_1s':float(np.mean(np.abs(ds)<=FPS)),
            'end_median':float(np.median(de)),'end_within_1s':float(np.mean(np.abs(de)<=FPS))}


def evaluate(all_cases,p=db.P,refine=True):
    rows={}
    for match,half,mode,pitch,ball,players,state,owner in all_cases:
        db.PITCH=pitch                          # ball_speed / on_restart_spot read the global
        try: dead=db.dead_mask(ball,players,p=p,refine=refine)
        finally: db.PITCH=(105.3,68.)
        rows.setdefault(mode,[]).append(dict(score(dead,state,owner),match=match,half=half))
    return rows


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--external-dir',required=True); a.add_argument('--cache',default='work/features')
    a.add_argument('--out',type=Path)
    args=a.parse_args(); data=list(cases(args.external_dir,args.cache)); report={}
    for refine in (True,False):
        for mode,rows in evaluate(data,refine=refine).items():
            report[f'{mode}_refine_{"on" if refine else "off"}']=summary(rows)
    print(json.dumps(report,indent=2))
    if args.out: args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
