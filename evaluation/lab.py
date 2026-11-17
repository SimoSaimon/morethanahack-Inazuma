"""Shared pieces for experiments on the external IDSSE matches (never on BAR-ATH labels).

    from evaluation.lab import Case
    c = Case('J03WPY', 'occluded'); c.score(c.fused())
"""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'third_party/metrica/evaluation'))
import possession_metrics as pm  # noqa: E402
from possession import dead_ball as db  # noqa: E402
from possession.augmentation import degrade  # noqa: E402
from possession.data import external_case  # noqa: E402
from possession.idsse import prepare  # noqa: E402
from possession.integration import fuse  # noqa: E402
from possession.models import load, sampled_score, sigmoid  # noqa: E402

EXT = ROOT / 'work/external'; CACHE = ROOT / 'work/features'; LAB = ROOT / 'work/lab'
DEV = ('J03WOH', 'J03WOY', 'J03WMX')      # the possession model was trained on these
VAL = 'J03WPY'; TEST = 'J03WQQ'
MODES = ('clean', 'occluded', 'severe')


def raw_match(match):
    with np.load(prepare(EXT / f'{match}_positions.xml', EXT / f'DFL-MAT-{match}_meta.xml', CACHE / 'raw')) as f:
        z = {k: f[k] for k in f.files}
    return z, json.loads(str(z['metadata']))['pitch_m']


def halves(match, mode):
    """(pitch, ball, players with speed, true state, true owner, degraded raw, teams) per half."""
    z, pitch = raw_match(match)
    for half in (0, 1):
        raw = z[f'raw{half}']; team = z[f'team{half}']
        xy = raw[:, :-1, :2] * np.asarray(pitch); speed = np.full(xy.shape[:2], np.nan)
        speed[2:-2] = np.linalg.norm(xy[4:] - xy[:-4], axis=-1) * db.FPS / 4
        players = np.concatenate([raw[:, :-1, :2], speed[..., None]], axis=2).astype(np.float64)
        deg = degrade(raw, team, pitch, match, half, mode)
        yield tuple(pitch), deg[:, -1, :].astype(np.float64), players, z[f'state{half}'], z[f'owner{half}'], deg, team


def detect(match, mode, p=db.P, refine=True):
    out = []
    for pitch, ball, players, *_ in halves(match, mode):
        db.PITCH = pitch
        try: dead = db.dead_mask(ball, players, p=p, refine=refine)
        finally: db.PITCH = (105.3, 68.0)
        out.append(np.where(dead, 0, 1).astype(np.int8))
    return np.concatenate(out)


def carried(owner):
    ok = owner >= 0; idx = np.where(ok, np.arange(len(owner)), 0); np.maximum.accumulate(idx, out=idx)
    idx[:np.argmax(ok)] = np.argmax(ok)
    return owner[idx].astype(np.int8)


class Case:
    def __init__(self, match, mode, model=None):
        self.match, self.mode = match, mode
        self.model = model or load(ROOT / 'models/possession_model.json')
        self.c = external_case(EXT, CACHE, match, mode)
        self.blocks = [tuple(map(int, b)) for b in self.c['blocks']]
        self.gt = np.where(self.c['state'] == 0, pm.D, carried(self.c['owner'])).astype(np.int8)
        self._state = None; self._logit = {}

    @property
    def state(self):
        if self._state is None:
            LAB.mkdir(parents=True, exist_ok=True); f = LAB / f'state-{self.match}-{self.mode}.npy'
            if f.exists(): self._state = np.load(f)
            else: self._state = detect(self.match, self.mode); np.save(f, self._state)
        return self._state

    def logit(self, stride=None):
        stride = stride or self.model['prediction_stride']
        if stride not in self._logit:
            self._logit[stride] = sampled_score(self.c['x'], self.c['blocks'], self.model, stride=stride)
        return self._logit[stride]

    def fused(self, logit=None, state=None, decoder=None):
        logit = self.logit() if logit is None else logit
        labels, _ = fuse(sigmoid(logit), self.c['quality'], self.state if state is None else state, self.blocks,
                         decoder or self.model['decoder'])
        return labels

    def score(self, pred):
        parts = lambda v: np.concatenate([v[self.blocks[0][0]:self.blocks[0][1]], [pm.X], v[self.blocks[1][0]:self.blocks[1][1]]]).astype(np.int8)
        r = pm.evaluate(parts(self.gt), parts(pred), bootstrap_n=0)
        return r['spell_f1_mean'], r['macro_f1'], r['n_gt_spells'], r['n_pred_spells']

