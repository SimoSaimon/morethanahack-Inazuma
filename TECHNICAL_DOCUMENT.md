# Inazuma — technical document

More Than A Hack 2027, Phase 1: one label per frame of the BAR-ATH tactical video
(`H` Barcelona in possession, `A` Athletic Club in possession, `D` dead ball, `X` no answer).

Code: <https://github.com/SimoSaimon/morethanahack-Inazuma>. Everything in this document can be traced to a file in that repository;
the [README](README.md) has the run instructions.

## 1. Approach

We split the label into two questions and answer each with its own component:

| Component | Question | Output |
|---|---|---|
| Game state (`pipeline/game_state.py`, `possession/dead_ball.py`) | Is the ball in play? | `in_play` / `stopped` per frame |
| Possession (`pipeline/run.py`, `possession/`) | If the ball is in play, which team controls it? | `P(H)` per frame, given live play |
| Fusion (`pipeline/fuse.py`, `possession/integration.py`) | Combine both into one coherent sequence | `H` / `A` / `D` / `X` per frame |

The split follows the ground truth's own definition (a frame is `D` when the ball is dead,
otherwise the owning team) and lets us train and test each part against the matching field of an
external dataset (`BallStatus` and `BallPossession`).

**Why this design.** The ranking metric (spellF1) scores whole possessions, so a per-frame
classifier is not enough: a single wrong frame splits a possession in two. We therefore (a) give
the model temporal context through its features, and (b) choose the final H/A sequence globally
with a Viterbi decoder that pays a cost for every change of team, instead of taking the most
likely team frame by frame. The ball is missing on ~41% of graded frames (mostly during
stoppages: ~19% of the frames we label as live), so the model is trained on data with
artificially removed and perturbed ball and player positions, and we never interpolate the
ball's coordinates during a gap.

**The ball is smoothed.** Comparing unlabelled ball kinematics, the BAR-ATH tracker smooths the
ball heavily: the velocity change over 3 frames has a 99th percentile of 3.7 m/s against 16.4 in
the external training data, so a touch is a gentle bend about a second long rather than a kink.
We therefore smooth the training ball the same way (Gaussian, sigma 6 frames, inside each visible
run; the value reproduces BAR-ATH's distribution of velocity changes) and add features that look
at longer spans.

**A sequence model.** Possession unfolds over time, so the model that reads the features is a small
temporal convolutional network that sees about 2.5 s on each side of every frame, instead of a
model that scores one frame at a time. On the held-out match the full pipeline went from 0.402
(trees, unsmoothed training ball) to 0.467 (trees, smoothed ball, new features) to 0.522 (network).

### 1.1 Possession model

1. **Parsing and cleaning.** Strict parsers for the EPTS tracking and XML metadata
   (`possession/io.py`); the speed channel is inconsistent, so speeds are recomputed from XY, and
   windows containing gaps or impossible jumps are discarded. A goalkeeper slot found within 25 m
   of the *other* goal line is filed under the wrong team (16 stretches in BAR-ATH, while its own
   team has nobody near its goal) and is moved to the other team (`possession/cleaning.py`).
2. **71 features per frame** (`possession/features.py`, `possession/flight.py`), computed per team and combined as
   differences plus shared context: distances of the two closest players of each team to the ball,
   density around the ball, relative velocity and approach, positions projected 0.4 s and 0.8 s
   ahead, team spread and speed, count of observations, and memory (1 s and 3 s windows, past and
   future evidence decayed over 3 s and 12 s, age of the last observation). Eighteen of them are
   built for a smoothed ball: which team came closest to the ball in the last and in the next
   1-2 s, kept apart; the ball's "flights" between two velocity changes measured over 0.8 s (which
   team stood at the start, which at the end, how far away in time); and whether the ball is being
   slowed or sped up. Tracking slots are not identities, so every feature is collective and
   invariant to slot order.
3. **Temporal convolutional network** (`training/train_tcn.py`). Input: the 71 features of every
   frame, standardised. A 1×1 convolution to 64 channels, then six residual blocks, each a
   3-tap convolution with dilation 1, 2, 4, 8, 16, 32 followed by a 1×1 convolution, then a 1×1
   output: one score per frame from a window of 127 frames (2.5 s each side). About 103,000
   weights per network; three networks trained from different seeds are averaged. It is trained
   with PyTorch on random 60-second excerpts (binary cross-entropy on live frames with a known
   team; half of the excerpts with the two teams swapped) and exported to JSON; inference is
   re-implemented in NumPy (`possession/models.py`), within 2.3e-6 of PyTorch. The model is made
   **team-symmetric by construction**: `score_H = [f(x) - f(x with teams swapped)] / 2`,
   `P(H) = sigmoid(score_H)`. Gradient-boosted trees on the same features (400 trees, 31 leaves)
   are trained too and compared on the validation match in exactly the same way; the network won
   (0.653 against 0.559) and is the model the pipeline runs.
4. **Smart Tagging prior** (challenge only). Offensive phases (`BUILD UP`, `PROGRESSION`,
   `CREATING CHANCES`, `ATTACKING TRANSITION`) push towards the tagged team, defensive phases
   towards its opponent. The correction is `0.6 × (tag_H − tag_A) × (1 − 0.75 × quality)`, added to
   the score before the sigmoid, so it weighs more when the tracking is poor. 0.6 is an engineering
   choice, not a fitted value.
5. **Temporal decoding.** Inside each stretch of live play, a two-state Viterbi path maximises the
   summed log-probability of the chosen team minus a switch cost per change of
   `0.2 × (1 + 2 × (1 − quality))`: 0.2 with full tracking, up to 0.6 when the ball is unseen.

### 1.2 Game-state component

A rule-based detector on the tactical tracking (`possession/dead_ball.py`), with three signals:

- **ball out:** the ball tracked outside the pitch rectangle starts a stoppage;
- **nobody runs:** the 90th-percentile player speed, averaged over 4 s, drops below 2.0 m/s
  (hysteresis: play resumes above 2.5 m/s). This catches fouls, where the ball never leaves the
  pitch; the mean speed was rejected because it flags slow build-up as dead. The two thresholds
  need no labels: that speed has two modes over a match, "nobody runs" and "playing", and the
  thresholds are the edges of the valley between them, found from the match itself
  (`speed_valley`; 2.0-2.5 m/s in both halves of BAR-ATH);
- **restart kick:** a still or unseen ball that is suddenly struck above 4.5 m/s ends a stoppage.

A state machine enters a stoppage on *out* or *slow*, keeps it until a restart, then removes dead
runs shorter than 4 s and fills live runs shorter than 1 s. Two refinements move the boundaries to
where the event happened, because spellF1 pays heavily for lateness (1 s late ≈ 0.65): a stoppage
starts where the players leave their pre-whistle peak speed, and ends at the restart kick (from the
set-piece spot when there is one). Smart Tagging kickoffs and goal kicks place their restart at the
tag start + 80 frames, snapped to the nearest detected kick. Before the opening kickoff of each
graded half the ball is dead; that kickoff is located the same way from its `Kick Off (Start)` tag
(or, without tags, as the first restart kick from the centre spot). No frame number of this match
appears in the code.

### 1.3 Fusion

`pipeline/fuse.py` checks that both components are on the same timeline (tactical video, 25 fps, frame 0
to 199318, exact row count and order, SHA-256 of the inputs and of the model) and then:

- `stopped` → `D`, and the decoder restarts from scratch after every stoppage;
- `in_play` → `H` or `A` from the Viterbi path;
- outside the two graded halves published by the organisers (`[11185, 90652)` and
  `[110808, 190198)`) → `X`, never scored.

The interface also accepts an `unknown` state (→ `X`), but the game-state component does not emit
it: around stoppages an `X` costs on both metrics (macroF1 counts it as a miss; spellF1 forward-fills
the previous team across the stoppage), so we always commit to an answer.

## 2. Assets used and dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (ATD TXT + XML) | Yes | Positions of ball and players, team of each slot, home/away |
| Smart Tagging | Yes | Phases as a soft prior on the team; set pieces to place restarts |
| Tactical homography | No | Not needed: the ATD is already in pitch coordinates |
| Tactical and broadcast video | No | Kept for future work (ball detection when the tracker loses it) |
| Broadcast tracking | No | We synchronised it to the tactical video from the two ball tracks (fixed offset per half; median ball disagreement 0.8 m). It fills only 23% of the live frames where the tactical ball is missing, and is more than 10 m off on 7% of shared frames, so we left it out |
| ATD speed channel | Partly | Possession recomputes speed from XY; the game-state detector uses the channel, whose values match m/s computed from positions |
| Smart Tagging free kicks, throw-ins, corners | Dropped for restarts | Their tag start does not predict when play resumes precisely enough |
| Public 5-minute ground truth | Only as a check | Frame alignment, file format and a final look at the finished pipeline; never to train or to choose a parameter |

## 3. Training data

| Dataset | Source | Labels | Amount |
|---|---|---|---|
| IDSSE (Bassek, Rein, Weber & Memmert, 2025, *Scientific Data* 12, 195; CC BY 4.0) | [huggingface.co/datasets/pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data) | `BallPossession` and `BallStatus` of the official DFL positional data. We did not label anything | 7 German professional matches |

- Possession model: train `J03WOH`, `J03WOY`, `J03WMX`, `J03WN1`, `J03WR9`; validation `J03WPY`
  (model and decoder selection); test `J03WQQ`, untouched until the model was frozen. Each match
  is used in three versions (original, occluded, degraded: ball and player gaps, 3 cm / 8 cm
  noise, slot swaps within a team), always with the ball smoothed as described above. The network
  reads whole sequences at 25 Hz; the trees are fitted on one live frame in five, with teams
  swapped (464,802 rows).
- Game-state component: no learning; its timing parameters were chosen against the IDSSE
  `BallStatus` of four matches with the ball occluded on ~40% of frames, `J03WQQ` held out
  (`evaluation/evaluate_dead_ball.py`).
- **No label of any kind about BAR-ATH** — human, event data or the private ground truth — is used
  anywhere. File hashes and the exact URLs are recorded in `models/external_sources.json`.

## 4. Possession rules implemented

- **Ball in play vs dead** comes only from the game-state component, never from the possession
  model: a player standing next to the ball before a throw-in does not make the ball live.
- **Controlled possession** is what the external `BallPossession` labels call possession; the
  model learns it from geometry and motion around the ball, not from a hand-written rule.
- **Loose balls, touches and deflections**: the memory features and the switch cost keep the
  previous team until the evidence for the other team is sustained. A single frame in which an
  opponent is closest to the ball is not enough. We impose no fixed minimum duration.
- **Ball not visible**: the decoder keeps the team through the gap unless the players' layout and
  the evidence on both sides of the gap say otherwise; an old observation weighs less and less.
- **Restarts**: after every `D` stretch the decision starts again from the new evidence (whoever
  plays the ball after the restart), with no carry-over of the team that had it before.
- **Dead-ball ownership** is not inferred: dead is always `D`.
- The model is offline: it uses frames after the one being labelled.

## 5. Results

All numbers are on the IDSSE match `J03WQQ`, which neither component saw while being built or
tuned, scored with the official Metrica scorer. Each value is given for three versions of the
tracking: clean / occluded (ball missing on 47% of live frames) / degraded (ball missing on 78%
of live frames, more noise); the ball is smoothed in all three. BAR-ATH, with the ball missing on
~19% of the frames we label as live, sits between the first two. IDSSE has no Smart Tagging, so
the tag prior is off throughout.

| Pipeline on `J03WQQ` | spellF1 | macroF1 |
|---|---|---|
| **Full pipeline: possession model + game-state detector** (`evaluation/evaluate_end_to_end.py`) | **0.522** (0.633 / 0.483 / 0.450) | 0.809 (0.845 / 0.801 / 0.780) |
| Nearest player + temporal smoothing, with the same game-state detector | 0.239 (0.386 / 0.207 / 0.124) | 0.694 |
| Possession model with the *true* game state (`evaluation/evaluate.py`) | 0.647 (0.740 / 0.607 / 0.593) | 0.929 |
| Game-state detector with the *true* possession (`evaluation/evaluate_end_to_end.py`) | 0.844 (0.870 / 0.838 / 0.823) | 0.871 |
| Trees instead of the network (same features, same five training matches), full pipeline | 0.467 (0.597 / 0.423 / 0.382) | 0.796 |
| First model (53 features, three matches, unsmoothed training ball), full pipeline, same smoothed tracking | 0.402 (0.498 / 0.364 / 0.343) | 0.781 |

Rows three and four isolate each component by replacing the other one with the truth; their
errors compound in the full pipeline. With the network the two components now cost about the
same: the possession model reaches 0.65 given the true state, and the full pipeline loses 0.125
to the game-state detector. Short possessions are still merged or missed: with the true state
the model produces 393 / 303 / 285 possessions against 401 real ones. The game-state detector
marks 30–33% of the IDSSE frames as stopped against a true 41%, i.e. it under-detects dead time
on this tracker. For the game-state component alone, re-timing on the other four matches raised
the occluded score from 0.815 to 0.837 (measured before the ball smoothing was added).

Things we tried on the validation match and dropped because they did not help: scoring the trees
on every frame instead of every fifth; other switch costs; a plain neural network that scores
one frame at a time (worse than the trees, so the gain of the network comes from reading time);
and a rule that gives the ball to the team of the last touch (velocity change next to a player),
which adds 0.08 on the unsmoothed external ball and nothing once the ball is smoothed like
BAR-ATH's.

The challenge match itself cannot be scored: the private ground truth is not available to
participants. The public 5-minute window (17 possessions) was used only
to verify that frames are aligned and labels mean what we think, and for one look at the finished
pipeline (spellF1 0.74, macroF1 0.95); at that sample size one missed possession moves spellF1 by
~6 points, so it is not a measure of quality, and no parameter was chosen on it.

## 6. Limitations and what we would do next

- **Short possessions** are still merged or missed, less than before. Next: longer training (the
  validation score was still rising at the last epoch), new simulated gaps at every epoch, a
  longer window, and a loss that weighs the frames around a change of possession.
- **Domain shift.** The model learned from DFL tracking, which is cleaner than Metrica's automatic
  tracking and may define possession slightly differently. We matched the ball smoothing but not
  other differences. Duplicate tracks of the same player are not merged, except goalkeepers
  filed under the wrong team.
- **Game state is hand-designed**, and it is now as costly as the possession model. Its two speed
  thresholds come from the unlabelled speed distribution of the match; on four IDSSE matches that
  choice scores 0.881 against 0.883 for the thresholds fitted to their labels. Next: learn the
  detector from IDSSE `BallStatus` with a sequence model like the possession one, and combine it
  with the rules that use Smart Tagging.
- **The Smart Tagging weight is not validated**, because no labelled data carries these tags.
- **One external test match.** The end-to-end figure comes from a single held-out IDSSE match with
  synthetic occlusions; Metrica's automatic tracking has different errors (fragmented and
  duplicated tracks, a ball lost in crowded areas rather than in random blocks).
- **Missing ball.** We do not recover the ball from video. The broadcast tracking loses the ball
  at much the same moments as the tactical one, so a ball detector on the tactical video is the
  remaining option; with the ball already seen on ~81% of live frames its upside is limited.

## 7. Reproduction

Python 3.12, NumPy 2.3.5 (inference) and pandas 2.2.3 (official validator). The submitted model and
file were produced on Linux with Python 3.14.10 and NumPy 2.5.4 (training with scikit-learn 1.9.1
and PyTorch 2.15.0 on CPU). No GPU, no PyTorch and no network access at run time.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-evaluation.txt
python make_submission.py --data-dir /path/to/data     # the folder that contains tactical/
```

`make_submission.py` runs `pipeline/run.py` (possession), `pipeline/game_state.py` (game state) and `pipeline/fuse.py`
(fusion), validates the result with the organisers' `validate_submission.py`, and writes
`results/Inazuma.csv` with its SHA-256. Re-running on the same data produces the same bytes.
Retraining and the external evaluation are optional and documented in the README
(`training/download_external.py`, `training/train.py`, `training/train_tcn.py`,
`evaluation/evaluate.py`, `evaluation/evaluate_dead_ball.py`, `evaluation/evaluate_end_to_end.py`).
Retraining the trees reproduces them byte for byte; the network's weights depend on the PyTorch
build and thread count, so the repository ships the trained network as JSON and the pipeline
runs that file. Automated tests: `python -W error -m unittest discover -s tests`.
