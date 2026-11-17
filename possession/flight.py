"""Possession features for a smoothed ball: who was near it before and after, and its flights.

The BAR-ATH tracker smooths the ball: velocity changes over 3 frames have quantiles
(50/90/99%) 0.8/2.2/3.7 m/s against 0.6/3.2/16.4 in IDSSE, so a touch is a gentle bend about a
second long rather than a kink. These features therefore look at longer spans:
  - which team came closest to the ball in the last and in the next 1-2 s, kept apart;
  - "flights", the ball path between two velocity changes measured over 0.8 s: which team stood
    at the start of the flight, which at its end, and how far away in time both are;
  - whether the ball is being slowed (received) or sped up (just played).
Difference features change sign under a team swap, context features do not."""
import numpy as np


def _team_nearest(raw, teams, pitch):
    scale = np.asarray(pitch, np.float64); teams = np.asarray(teams)
    ball = raw[:, -1, :2].astype(np.float64) * scale; xy = raw[:, :-1, :2].astype(np.float64) * scale
    with np.errstate(invalid='ignore'):
        d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    return ball, d[:, teams == 0].min(axis=1), d[:, teams == 1].min(axis=1)


def _window_min(v, w, future):
    """min of v over the next (future) or last w frames, current frame included."""
    out = v.copy()
    for s in range(1, w + 1):
        if future: out[:-s] = np.minimum(out[:-s], v[s:])
        else: out[s:] = np.minimum(out[s:], v[:-s])
    return out


def _window_mean(v, w, future):
    c = np.concatenate([[0.], np.cumsum(v)]); n = len(v); i = np.arange(n)
    lo, hi = (i, np.minimum(n, i + w + 1)) if future else (np.maximum(0, i - w), i + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def flight_features(raw, teams, pitch=(105.3, 68.0), fps=25, span=10, win=6, kick=1.5):
    ball, dh, da = _team_nearest(raw, teams, pitch); n = len(ball); seen = np.isfinite(ball[:, 0])
    cap = lambda v: np.minimum(v, 25.)
    geom = np.where(seen & np.isfinite(dh) & np.isfinite(da), np.clip((cap(da) - cap(dh)) / 2.5, -3, 3), 0.)
    names = []; cols = []; odd = []
    def add(name, col, is_odd): names.append(name); cols.append(np.nan_to_num(col, nan=0, posinf=0, neginf=0)); odd.append(is_odd)
    # A. who got closest to the ball just before / just after
    for sec in (1, 2):
        w = sec * fps
        for side, fut in (('past', False), ('future', True)):
            mh = _window_min(np.where(seen, dh, np.inf), w, fut); ma = _window_min(np.where(seen, da, np.inf), w, fut)
            ok = np.isfinite(mh) & np.isfinite(ma)
            add(f'x_min_{side}{sec}', np.where(ok, np.clip((cap(ma) - cap(mh)) / 2.5, -3, 3), 0), True)
            add(f'x_geom_{side}{sec}', _window_mean(geom, w, fut), True)
    # B. flights: the ball path between two velocity changes (acceleration peaks)
    step = (ball[span:] - ball[:-span]) * fps / span
    change = np.full(n, -1.); change[span:-span] = np.linalg.norm(step[span:] - step[:-span], axis=1); change[np.isnan(change)] = -1
    peak = change > kick
    for s in range(1, win + 1):
        peak[s:] &= change[s:] >= change[:-s]; peak[:-s] &= change[:-s] > change[s:]
    ix = np.arange(n)
    gap_id = np.cumsum(~seen)                                  # constant inside a visible stretch
    prev = np.maximum.accumulate(np.where(peak, ix, -1)); nxt = np.minimum.accumulate(np.where(peak, ix, n)[::-1])[::-1]
    for side, idx in (('prev', prev), ('next', nxt)):
        safe = idx.clip(0, n - 1); ok = (idx >= 0) & (idx < n) & seen & (gap_id[safe] == gap_id)
        age = np.abs(ix - safe) / fps
        add(f'x_peak_{side}_geom', np.where(ok, geom[safe], 0), True)
        add(f'x_peak_{side}_near', np.where(ok, np.minimum(np.minimum(dh, da)[safe], 10) / 5, 2), False)
        add(f'x_peak_{side}_age', np.where(ok, np.minimum(age, 10) / 5, 2), False)
        add(f'x_peak_{side}_size', np.where(ok, np.minimum(change[safe], 20) / 10, 0), False)
    # C. speed trend: a ball being slowed is being received, one speeding up was just played
    speed = np.full(n, np.nan); speed[span:] = np.linalg.norm(step, axis=1)
    trend = np.full(n, 0.); trend[span:-span] = speed[2 * span:] - speed[span:-span]
    add('x_speed_trend', np.clip(trend / 10, -3, 3), False)
    add('x_change', np.clip(change, 0, 20) / 10, False)
    return np.column_stack(cols).astype(np.float32), names, [-1 if o else 1 for o in odd]
