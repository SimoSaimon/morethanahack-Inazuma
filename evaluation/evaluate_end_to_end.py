"""End-to-end check on the held-out IDSSE match: both components together, no oracle.

The two component tests each replace the other component with the truth: evaluation/evaluate.py gives the
possession model the true game state, evaluation/evaluate_dead_ball.py gives the game-state detector the
true possession. Here both run together on the same degraded tracking of J03WQQ, a match neither
component saw while being built or tuned, and the result is scored by the official Metrica scorer
exactly like a submission (macroF1 and spellF1 over H/A/D). IDSSE has no Smart Tagging, so the
tag prior and the tag-based restarts are off.

    python evaluation/evaluate_end_to_end.py --external-dir work/external --cache work/features
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
from possession.augmentation import degrade
from possession.data import external_case
from possession.idsse import prepare
from possession.integration import fuse
from possession.models import load, sampled_score, sigmoid
from possession.temporal import decode
sys.path.insert(0, str(ROOT / 'third_party/metrica/evaluation'))
import possession_metrics as pm  # noqa: E402

MATCH = 'J03WQQ'
MODES = ('clean', 'occluded', 'severe')


def detector_states(z, pitch, mode):
    """1 in play / 0 stopped per frame, halves concatenated as in external_case.

    The ball is degraded with the same seed as the possession features, so both components see
    the same gaps. Player speeds come from the clean positions, as in evaluation/evaluate_dead_ball.py,
    because the BAR-ATH tracker supplies speed directly."""
    out = []
    for half in (0, 1):
        raw = z[f'raw{half}']; team = z[f'team{half}']
        xy = raw[:, :-1, :2] * np.asarray(pitch); speed = np.full(xy.shape[:2], np.nan)
        speed[2:-2] = np.linalg.norm(xy[4:] - xy[:-4], axis=-1) * db.FPS / 4
        players = np.concatenate([raw[:, :-1, :2], speed[..., None]], axis=2).astype(np.float64)
        ball = degrade(raw, team, pitch, MATCH, half, mode)[:, -1, :].astype(np.float64)
        db.PITCH = tuple(pitch)            # ball_speed / on_restart_spot read the global
        try:
            dead = db.dead_mask(ball, players)
        finally:
            db.PITCH = (105.3, 68.0)
        out.append(np.where(dead, 0, 1).astype(np.int8))
    return np.concatenate(out)


def carried_owner(owner):
    """The recorded owner, carried over the rare frames without one (head back-filled)."""
    ok = owner >= 0
    idx = np.where(ok, np.arange(len(owner)), 0); np.maximum.accumulate(idx, out=idx)
    idx[:np.argmax(ok)] = np.argmax(ok)
    return owner[idx].astype(np.int8)


def truth(state, owner):
    """H/A/D codes: D where the ball is dead, else the owning team (no fourth state, as in the
    challenge truth)."""
    return np.where(state == 0, pm.D, carried_owner(owner)).astype(np.int8)


def match_codes(codes, blocks):
    """One array per match with an X frame between the halves, so no possession spans halftime."""
    parts = [codes[a:b] for a, b in blocks]
    return np.concatenate([parts[0], [pm.X], parts[1]]).astype(np.int8)


def score(gt, pred, blocks):
    r = pm.evaluate(match_codes(gt, blocks), match_codes(pred, blocks), bootstrap_n=0)
    return {k: r[k] for k in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells')}


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--external-dir', required=True, type=Path)
    a.add_argument('--cache', default='work/features', type=Path)
    a.add_argument('--out', type=Path)
    args = a.parse_args()
    model = load(ROOT / 'models/possession_model.json')
    baseline = json.loads((ROOT / 'models/training_report.json').read_text())['validation']['nearest_temporal']
    with np.load(prepare(args.external_dir / f'{MATCH}_positions.xml', args.external_dir / f'DFL-MAT-{MATCH}_meta.xml',
                         args.cache / 'raw')) as f:
        z = {k: f[k] for k in f.files}
    pitch = json.loads(str(z['metadata']))['pitch_m']
    report = {'match': MATCH, 'scorer': 'official possession_metrics.evaluate', 'smart_tagging': False, 'modes': {}}
    for mode in MODES:
        case = external_case(args.external_dir, args.cache, MATCH, mode)
        blocks = [tuple(map(int, b)) for b in case['blocks']]
        gt = truth(case['state'], case['owner'])
        state = detector_states(z, pitch, mode)
        if len(state) != len(gt):
            raise ValueError('Detector and features are not on the same frames')
        logit = sampled_score(case['x'], case['blocks'], model, stride=model['prediction_stride'])
        ours, _ = fuse(sigmoid(logit), case['quality'], state, blocks, model['decoder'])
        oracle_state, _ = fuse(sigmoid(logit), case['quality'], case['state'], blocks, model['decoder'])
        near, _ = decode(case['nearest_logit'].astype(float), case['quality'], state, blocks, fps=25,
                         switch_cost=baseline['switch_cost'], gap_penalty=baseline['gap_penalty'])
        # as in evaluation/evaluate_dead_ball.py: the true (carried) owner wherever the detector says alive
        oracle_owner = np.where(state == 0, pm.D, carried_owner(case['owner'])).astype(np.int8)
        report['modes'][mode] = {
            'end_to_end': score(gt, ours, blocks),
            'possession_with_true_state': score(gt, oracle_state, blocks),
            'game_state_with_true_possession': score(gt, oracle_owner, blocks),
            'nearest_player_baseline_end_to_end': score(gt, near, blocks),
            'stopped_share_true': float((case['state'] == 0).mean()),
            'stopped_share_detected': float((state == 0).mean()),
        }
        print(mode, json.dumps(report['modes'][mode]['end_to_end']), flush=True)
    keys = ('end_to_end', 'possession_with_true_state', 'game_state_with_true_possession', 'nearest_player_baseline_end_to_end')
    report['mean_over_modes'] = {k: {m: float(np.mean([report['modes'][x][k][m] for x in MODES]))
                                     for m in ('spell_f1_mean', 'macro_f1')} for k in keys}
    print(json.dumps(report['mean_over_modes'], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
