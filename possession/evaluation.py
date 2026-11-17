from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent.parent/'third_party/metrica/evaluation'))
import possession_metrics as official


def assess(case,pred,probability=None):
    valid=(case['state']==1)&(case['owner']>=0)
    gt=np.where(case['state']==0,2,np.where(valid,case['owner'],3)).astype(np.int8)
    # Respect the two source halves even when stored contiguously.
    gt_spells=[]; pred_spells=[]
    for a,b in case['blocks']:
        block_runs=official.valid_runs(gt[a:b])
        # Fill external unknown ownership as the official scorer does with X.
        filled=official.fill_abstentions(pred[a:b],block_runs)
        gt_spells.extend([(s+a,e+a,k) for s,e,k in official.possession_spells(gt[a:b],block_runs)])
        pred_spells.extend([(s+a,e+a,k) for s,e,k in official.possession_spells(filled,block_runs)])
    overlap=official.spell_overlaps(gt_spells,pred_spells); scores={}
    for threshold in official.SPELL_IOUS:
        tp,fp,fn=official.match_spells(overlap,len(gt_spells),len(pred_spells),threshold)
        scores[str(threshold)]=2*tp/max(1,2*tp+fp+fn)
    perclass=[]
    for k in (0,1):
        tp=np.sum(valid&(gt==k)&(pred==k)); fp=np.sum(valid&(gt!=k)&(pred==k)); fn=np.sum(valid&(gt==k)&(pred!=k))
        perclass.append(float(2*tp/max(1,2*tp+fp+fn)))
    report={'spellF1_with_oracle_state':float(np.mean(list(scores.values()))),'spellF1_per_iou':scores,
            'team_accuracy_on_live':float(np.mean(pred[valid]==gt[valid])),'team_macroF1_on_live':float(np.mean(perclass)),
            'n_gt_spells':len(gt_spells),'n_pred_spells':len(pred_spells),'live_frames':int(valid.sum()),
            'ball_seen_fraction_live':float(case['ball_seen'][valid].mean())}
    if probability is not None:
        report['brier_home_on_live']=float(np.mean((probability[valid]-(gt[valid]==0))**2))
    for name,mask in [('visible',valid&case['ball_seen']),('missing',valid&~case['ball_seen'])]:
        report[f'team_accuracy_{name}']=float(np.mean(pred[mask]==gt[mask])) if mask.any() else None
    return report
