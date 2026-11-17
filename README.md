# Inazuma — ball possession per frame (More Than A Hack 2027, Phase 1)

For each of the 199,319 frames of the BAR-ATH tactical video this code writes one label:
`H` Barcelona in possession, `A` Athletic Club in possession, `D` ball not in play, `X` no answer.

Two components, fused into one sequence (details in [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md)):

- **Possession** (`pipeline/run.py`, `possession/`): a temporal convolutional network trained on five
  external Bundesliga matches (IDSSE, CC BY 4.0) gives P(H) per frame, plus a small Smart Tagging hint.
- **Game state** (`pipeline/game_state.py`, `possession/dead_ball.py`): rule-based detector of stoppages.
- **Fusion** (`pipeline/fuse.py`): Viterbi decoding that pays a cost for every change of team.

## Reproduce the submission file

Python 3.12 and the pinned packages. No GPU, no network access and no PyTorch are needed: the trained
network ships as JSON in `models/`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-evaluation.txt        # numpy 2.3.5 + pandas 2.2.3 (official validator)
python make_submission.py --data-dir /path/to/data
```

`/path/to/data` is the folder that **contains** `tactical/`, with the challenge files
`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_smart_tagging.xml`.
The match data is not part of this repository and must be obtained from the organisers.

The command runs possession, game state and fusion, validates the result with the organisers'
`validate_submission.py` and writes to `results/`:

- `Inazuma.csv`: the submission;
- `Inazuma.md`: a copy of the technical document;
- intermediate files, and the SHA-256 of the two files above.

Expected SHA-256 of `Inazuma.csv`:

```text
01a3d94c41abecf88d7932fc44b6264ad3493fae9ab76f73de1963514b436210
```

The same bytes were obtained on Python 3.12.15 / NumPy 2.3.5 (macOS, arm64) and on
Python 3.14 / NumPy 2.5.4 (Linux). `--no-tags` drops Smart Tagging and is not expected to give the same file.

## Tests

```bash
python -W error -m unittest discover -s tests
```

## Layout

```text
make_submission.py   one command from the challenge data to the files to upload
pipeline/            run.py (possession), game_state.py (game state), fuse.py (fusion)
possession/          features, model inference, dead-ball rules, decoding
models/              trained model (JSON) and its training reports
third_party/metrica/ unmodified scorer files from the organisers' repository (MIT, see PROVENANCE.md)
tests/               automated tests (no match data needed)
training/            optional: download external data and retrain the models
evaluation/          optional: evaluation on the external matches
```

## Optional: retrain and evaluate

About 2.6 GB of external data is downloaded. The submission does not need any of this.

```bash
pip install -r requirements-training.txt
python training/download_external.py --out work/external
python training/train.py     --external-dir work/external --cache work/features --out work/retrained
python training/train_tcn.py --external-dir work/external --cache work/features --out work/retrained
python evaluation/evaluate.py --external-dir work/external --cache work/features --models work/retrained --out work/test_report.json
```

Retraining the trees reproduces them byte for byte. The network's weights depend on the PyTorch build
and thread count, so the repository ships the trained network and the pipeline runs that file.

## Data and licences

- Challenge match data (tracking, video, Smart Tagging) is **not** included and must not be redistributed.
- External training data: IDSSE, Bassek, Rein, Weber & Memmert (2025), *An integrated dataset of
  spatiotemporal and event data in elite soccer*, Scientific Data 12, 195
  ([paper](https://doi.org/10.1038/s41597-025-04505-y),
  [data](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0). The model weights derive
  from it through the scripts in `training/`.
- `third_party/metrica/`: scorer files from the organisers' repository under the MIT licence;
  the licence does not cover the match data.
