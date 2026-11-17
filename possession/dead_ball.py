"""Ball in play or not, per video frame, from the tactical ATD alone.

Three signals:

  out    ball tracked outside the pitch rectangle. ~98% precise, but rare: it marks the moment
         the ball leaves, after which it is carried back in or lost by the tracker.
  slow   nobody runs. The 90th-percentile player speed over 3 s drops from ~3.7 m/s in play to
         ~1.8 m/s at a stoppage, and it catches fouls, where the ball never leaves the pitch. The
         mean speed was tried first and flags slow build-up as dead; "nobody runs" does not.
  kick   the ball suddenly moving fast. Used to end a stoppage: play restarts with a kick.

State machine: enter dead on `out` or on `slow` (with hysteresis), stay dead until the restart,
then clean up runs too short to be real.

    python pipeline/game_state.py --data-dir ... --out results   # writes game_state.csv for pipeline/fuse.py
"""

import warnings

import numpy as np

FPS = 25
PITCH = (105.3, 68.0)   # BAR-ATH real pitch in metres (Metrica games.py); the XML says 105x68

# Timing parameters tuned on the DFL BallStatus of four IDSSE matches with the ball occluded
# ~40% (evaluation/evaluate_dead_ball.py; J03WQQ held out: oracle spellF1 0.815 -> 0.837).
# The two speed thresholds need no labels: the smoothed p90 player speed of a match has two modes,
# "nobody runs" (~1.5 m/s) and "playing" (~3.3 m/s), and the hysteresis brackets the valley
# between them (speed_valley). On BAR-ATH the valley is 2.0-2.5 m/s in both halves, on the IDSSE
# matches 2.0-2.25/2.5, so 2.0/2.5 below is that valley and pipeline/game_state.py re-derives it from the
# match it is given. No BAR-ATH possession label enters these numbers.
# restart_back stays at 150 rather than 200, the more conservative of the two values tried.
P = dict(
    out_margin=0.0,      # pitch units beyond the line before the ball counts as out
    slow_window=100,     # frames of centred mean for player speed (4 s)
    slow_enter=2.0,      # m/s p90 player speed: below this nobody is running (lower edge of the valley)
    slow_exit=2.5,       # m/s: above this they are playing again (upper edge of the valley)
    kick_speed=4.5,      # m/s ball speed that counts as a kick
    out_max=15 * FPS,    # an out-of-play stretch is closed by a kick within this many frames
    quiet=8,             # frames of still-or-unseen ball before a kick that counts as a restart
    ramp=12,             # frames the tracked ball may take to get from still to kick speed
    react=0,             # players start running this many frames after the restart kick
    restart_back=150,    # how far back from the detected end to look for the restart
    tag_lead=80,         # frames from a set-piece tag's start to the restart it marks
    tag_snap=20,         # snap to a restart kick within this many frames: ~2x the 69-88 spread
    stop_back=60,        # how far back from a detected start to look for the whistle
    stop_drop=0.5,       # m/s the players must have slowed by for a start to be moved back
    stop_peak=0.6,       # m/s below the pre-whistle peak that still counts as "still running"
    stop_ball=4.0,       # m/s: a ball this fast means play was still on, never move a start past it

    centre_radius=3.0,   # metres from the centre spot for a kick to count as a kickoff (no tags)

    min_dead=4 * FPS,    # dead runs shorter than this are dropped
    min_alive=1 * FPS,   # alive runs shorter than this, between two dead ones, are filled
)


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def centred_mean(v, window):
    """Centred rolling mean ignoring NaN."""
    ok = ~np.isnan(v)
    c = np.r_[0, np.cumsum(np.where(ok, v, 0))]
    k = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(v))
    lo, hi = np.maximum(0, i - window // 2), np.minimum(len(v), i + window // 2 + 1)
    with np.errstate(invalid="ignore"):
        return (c[hi] - c[lo]) / (k[hi] - k[lo])


def speed_valley(p90, window=P["slow_window"], width=0.25, tolerance=1.1, split=2.5):
    """(enter, exit) for the `slow` hysteresis, from unlabelled player speeds alone.

    Histogram of the smoothed p90 speed in `width` m/s bins; the two modes are the highest bin
    below and above `split`; the valley is the lowest bin between them, widened to neighbours
    within `tolerance` of it. Returns the valley's edges."""
    v = centred_mean(np.asarray(p90, dtype=np.float64), window); v = v[~np.isnan(v)]
    edges = np.arange(0, 8 + width, width); h, _ = np.histogram(v, edges)
    smooth = np.convolve(h, [.25, .5, .25], "same"); cut = int(split / width)
    low = int(np.argmax(smooth[:cut])); high = cut + int(np.argmax(smooth[cut:]))
    k = low + int(np.argmin(h[low:high + 1])); a = b = k
    while a - 1 > low and h[a - 1] <= tolerance * h[k]: a -= 1
    while b + 1 < high and h[b + 1] <= tolerance * h[k]: b += 1
    return float(edges[a]), float(edges[b + 1])


def hysteresis(v, enter, exit_):
    """True from v < enter until v > exit_. NaN keeps the current state."""
    out = np.zeros(len(v), bool)
    state = False
    for i, x in enumerate(v):
        if state and x > exit_:
            state = False
        elif not state and x < enter:
            state = True
        out[i] = state
    return out


def ball_speed(ball):
    """m/s over a 4-frame baseline, NaN where the ball is missing at either end."""
    xy = ball[:, :2] * PITCH
    s = np.full(len(xy), np.nan)
    s[2:-2] = np.linalg.norm(xy[4:] - xy[:-4], axis=1) * FPS / 4
    return s


def restart_kicks(ball, out, p=P):
    """Frames where a still or unseen ball is suddenly struck: what a restart looks like.

    A restart has one of two shapes in the tracking: the ball is missing until the kick, so the
    tracker picks it up already moving, or it stands still and then jumps above `kick_speed`.
    The rule uses only the ball's own speed; no event time of the match enters it.
    """
    bs = ball_speed(ball)
    with np.errstate(invalid="ignore"):
        quiet = np.isnan(bs) | (bs < 3)
        fast = (bs > p["kick_speed"]) & ~out
    # The tracked ball takes up to ~10 frames to reach full speed after a kick, so the quiet
    # stretch is looked for up to `ramp` frames before the fast frame, and the kick is dated to
    # the first frame after it.
    c = np.r_[0, np.cumsum(quiet)]
    q, ramp = p["quiet"], p["ramp"]
    kicks = set()
    for t in np.where(fast)[0]:
        for back in range(0, ramp + 1):
            a = t - back - q
            if a >= 0 and c[t - back] - c[a] == q:
                kicks.add(t - back)
                break
    return np.array(sorted(kicks), int)


def on_restart_spot(ball):
    """Ball where a set piece is taken from: the goal area (goal kick), a corner, a touchline
    (throw-in). In metres, with a metre or so of slack for tracking error."""
    L, W = PITCH
    x, y = ball[:, 0] * L, ball[:, 1] * W
    with np.errstate(invalid="ignore"):
        end_line = (x < 6.5) | (x > L - 6.5)
        goal_area = end_line & (np.abs(y - W / 2) < 10.5)
        corner = ((x < 2) | (x > L - 2)) & ((y < 2) | (y > W - 2))
        touchline = (y < 1.5) | (y > W - 1.5)
    return goal_area | corner | touchline


def refine_restarts(dead, p90, kicks, spot_kicks=(), p=P):
    """Move each stoppage's end from where the smoothed speed says to the restart kick.

    The 3 s smoothing ends a stoppage 1-5 s late, and spellF1 pays heavily for that (1 s late is
    0.65). Find when the players actually start running, then the first restart kick shortly
    before it; without a kick, assume they react `react` frames after one.
    """
    fast = centred_mean(p90, 13)
    dead = dead.copy()
    spot_kicks = np.asarray(spot_kicks, int)
    for start, end in runs(dead):
        # A set piece from its spot is the restart, whatever the players do: at a goal kick or a
        # throw-in nobody sprints, so the "players start running" onset below never comes.
        k = spot_kicks[(spot_kicks > start + p["min_dead"]) & (spot_kicks <= end + 50)]
        if len(k):
            dead[k[0]:end] = False
            dead[end:k[0]] = True
            continue
        # refine_stops may have moved `start` back to where players still run flat out, so the
        # onset search must not begin before the stoppage has lasted min_dead.
        floor = min(start + p["min_dead"], end)
        lo = max(floor, end - p["restart_back"])
        onset = end
        for t in range(lo, end):  # first moment the players run for a whole second
            if np.all(fast[t:t + FPS] > p["slow_enter"]):
                onset = t
                break
        j = np.searchsorted(kicks, max(floor, onset - 60))
        new_end = kicks[j] if j < len(kicks) and kicks[j] <= onset + 5 else onset - p["react"]
        new_end = max(new_end, floor)
        dead[new_end:end] = False
        dead[end:new_end] = True
    return dead


def refine_stops(dead, out, p90, bs, p=P):
    """Move a foul's start back from where the smoothed speed says to the whistle.

    The 3 s smoothing only drops below threshold 2-4 s after play stops. At the whistle the
    players are still running flat out and then decelerate together, so the stoppage starts
    where the speed leaves its pre-whistle peak. A ball going out already starts on time, and a
    ball still being struck hard means the players slowed down with play still on.
    """
    fast = centred_mean(p90, 13)
    with np.errstate(invalid="ignore"):
        struck = bs > p["stop_ball"]
    dead = dead.copy()
    for start, _ in runs(dead):
        if out[max(0, start - 5):start + 5].any():
            continue
        lo = max(0, start - p["stop_back"])
        window = fast[lo:start + 1]
        if np.all(np.isnan(window)):
            continue
        peak = np.nanmax(window)
        if peak - fast[start] < p["stop_drop"]:
            continue                                   # no real deceleration: leave it
        new_start = lo + np.where(window >= peak - p["stop_peak"])[0][-1]
        hard = np.where(struck[new_start:start])[0]
        if len(hard):
            new_start += hard[-1] + 1
        dead[new_start:start] = True
    return dead


def signals(ball, players, p=P):
    x, y = ball[:, 0], ball[:, 1]
    m = p["out_margin"]
    with np.errstate(invalid="ignore"):
        out = (x < -m) | (x > 1 + m) | (y < -m) | (y > 1 + m)
    with warnings.catch_warnings():  # all-NaN frames (halftime) warn
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanpercentile(players[:, :, 2], 90, axis=1)
    slow = hysteresis(centred_mean(speed, p["slow_window"]), p["slow_enter"], p["slow_exit"])
    with np.errstate(invalid="ignore"):
        kick = (ball_speed(ball) > p["kick_speed"]) & ~out
    return out, slow, kick, speed


def set_piece_tags(events):
    """(start frame, type) of the Smart Tagging set pieces, free kicks left out.

    `events` is possession.io.load_tags(). Only kickoffs lead the restart by a roughly fixed time
    (about 80 frames; our own kick detector puts the kickoffs after a goal at 77-85). Throw-ins,
    corners and goal kicks do not: by two independent measures over the whole match (end of the
    detected stoppage, first kick from the restart spot) they restart a median 117-166 frames
    after the tag start, spread over ~70, so a fixed lead does not fit them. Free kicks are left
    out: their tag can start far before the kick, so it says nothing precise about when play
    resumes.
    """
    return [(e["start"], e["tags"].get("Type")) for e in events
            if e["code"] == "SET PIECES" and e["tags"].get("Type") != "Free Kick"]


def tag_restart_frame(tag, kicks, p=P):
    """The restart a set-piece tag marks: its start + tag_lead, snapped to the nearest restart
    kick within tag_snap frames."""
    guess = tag + p["tag_lead"]
    near = kicks[np.abs(kicks - guess) <= p["tag_snap"]]
    return int(near[np.argmin(np.abs(near - guess))]) if len(near) else int(guess)


def half_kickoffs(ball, kicks, tags, blocks, p=P):
    """Frame of the opening kickoff of each graded block (one block per half), or None.

    `blocks` are the graded halves published by the organisers. With Smart Tagging, the kickoff
    is the restart marked by the block's `Kick Off (Start)` tag, located like every other tag.
    Without it, the first restart kick struck within `centre_radius` of the centre spot. Nothing
    here is a frame number of the match: both rules work on any game.
    """
    L, W = PITCH
    with np.errstate(invalid="ignore"):
        centre = np.hypot((ball[:, 0] - 0.5) * L, (ball[:, 1] - 0.5) * W) < p["centre_radius"]
    found = []
    for a, b in blocks:
        starts = [t for t, ty in tags if ty == "Kick Off (Start)"
                  and a - p["tag_snap"] <= t + p["tag_lead"] < b]
        if starts:
            found.append(tag_restart_frame(min(starts), kicks, p))
            continue
        k = [int(t) for t in kicks if a <= t < b and (centre[max(t - 3, 0)] or centre[t])]
        found.append(k[0] if k else None)
    return found


def tag_restarts(dead, kicks, tags, p=P):
    """Put the restart where the tag says, snapped to the nearest restart kick.

    The stoppage before it ends there; if we had not seen a stoppage at all (a quick throw-in),
    one is opened from the tag's start, which is ~3 s before the restart.
    """
    dead = dead.copy()
    for tag in tags:
        restart = tag_restart_frame(tag, kicks, p)
        # the stoppage this restart closes: the dead run touching [tag, restart + snap]
        span = slice(tag, restart + p["tag_snap"])
        hit = [(a, b) for a, b in runs(dead) if a < span.stop and b > span.start]
        if hit:
            a, b = hit[0]
            if a >= restart:
                continue              # a later stoppage the tag does not close: leave it whole
            restart = max(restart, a + p["min_dead"])   # never leave a stoppage under min_dead
            dead[a:restart] = True
            dead[restart:max(b, restart)] = False
        else:
            dead[tag:restart] = True
    return dead


def dead_mask(ball, players, p=P, refine=True, tags=(), blocks=()):
    out, slow, kick, p90 = signals(ball, players, p)
    dead = slow.copy()

    # Ball out: dead from the exit until the first kick once it is back in.
    # A real restart (struck from rest or unseen) closes a ball-out stretch when the stretch
    # lasts a second or a set-piece tag backs it up: the ball re-entering the tracker at speed,
    # or coming back from a ball boy, is not one. A brief unconfirmed excursion may be tracker
    # noise, so it keeps the old rule (first fast ball).
    raw, rk = np.where(kick)[0], restart_kicks(ball, out, p)
    tag_end = np.array([t for t, _ in tags], int) + p["tag_lead"]
    for start, end in runs(out):
        confirmed = end - start >= FPS or ((tag_end > start) & (tag_end < end + 40 * FPS)).any()
        kicks = rk if confirmed else raw
        j = np.searchsorted(kicks, end)
        stop = kicks[j] if j < len(kicks) and kicks[j] - end < p["out_max"] else end
        dead[start:stop] = True

    for start, end in runs(~dead):
        if end - start < p["min_alive"] and start > 0 and end < len(dead):
            dead[start:end] = True
    for start, end in runs(dead):
        if end - start < p["min_dead"]:
            dead[start:end] = False
    if refine:
        dead = refine_stops(dead, out, p90, ball_speed(ball), p)
        kicks = restart_kicks(ball, out, p)
        spot = on_restart_spot(ball)
        spot_kicks = kicks[spot[np.maximum(kicks - 3, 0)] | spot[kicks]]
        dead = refine_restarts(dead, p90, kicks, spot_kicks, p)
    # Kickoffs and goal kicks only: the one goal kick known to the frame ends +11 with its tag
    # and -31 without. Throw-ins and corners restart well after tag + 80 (see set_piece_tags).
    kickoffs = [t for t, ty in tags if str(ty).startswith("Kick Off") or ty == "Goal Kick"]
    if kickoffs:
        dead = tag_restarts(dead, restart_kicks(ball, out, p), kickoffs, p)
    # A graded half can open before its kickoff (the 2nd half does, by ~2 min): until the
    # kickoff the ball is dead, whatever the players do while they line up.
    for (a, _), k in zip(blocks, half_kickoffs(ball, restart_kicks(ball, out, p), tags, blocks, p)):
        if k is not None and k - a >= p["min_dead"]:
            dead[a:k] = True
    return dead
