"""Tracking clean-up that needs no labels: goalkeepers filed under the wrong team.

In BAR-ATH the away goalkeeper's slot spends ~10% of the graded frames inside the home team's
defensive zone, while the away team has nobody within 45 m of its own goal: it is the home
goalkeeper filed under the wrong team. Left alone, a build-up from the goalkeeper looks like an
opponent standing on the ball.

Rule, from positions only: a team's goalkeeper is the slot that lives closest to the goal that
team defends. Whenever that slot is within `zone` metres of the *other* goal line, its
observations are moved to a slot of the other team that is empty in those frames.
"""
import numpy as np

ZONE = 25.0   # metres from the wrong goal line; a goalkeeper has no business there


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fix_goalkeeper_team(raw, teams, pitch=(105.3, 68.0), zone=ZONE):
    """Return (copy of raw with misfiled goalkeeper frames moved to the other team, report).

    `raw` is one period (teams do not change ends inside it): (frames, slots + ball, 3).
    """
    raw = np.array(raw, copy=True); teams = np.asarray(teams); x = raw[:, :-1, 0].astype(np.float64)
    report = []
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            side = [np.nanmedian(x[:, teams == t]) for t in (0, 1)]
            if not np.all(np.isfinite(side)) or (side[0] < .5) == (side[1] < .5): return raw, report
            for t in (0, 1):
                own = 0.0 if side[t] < side[1 - t] else 1.0
                idx = np.flatnonzero(teams == t)
                depth = np.nanmedian(np.abs(x[:, idx] - own), axis=0)
                if not np.isfinite(depth).any(): continue
                keeper = idx[np.nanargmin(depth)]
                wrong = np.abs(x[:, keeper] - (1 - own)) * pitch[0] < zone   # NaN compares False
                for a, b in runs(wrong):
                    free = [s for s in np.flatnonzero(teams == 1 - t) if np.isnan(raw[a:b, s, 0]).all()]
                    if not free: continue
                    raw[a:b, free[0]] = raw[a:b, keeper]; raw[a:b, keeper] = np.nan
                    report.append({'slot': int(keeper) + 1, 'to_slot': int(free[0]) + 1, 'start': int(a), 'end': int(b)})
    return raw, report
