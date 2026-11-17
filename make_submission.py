"""One command from the challenge data bundle to the files we upload.

    python make_submission.py --data-dir /path/to/data

`--data-dir` is the folder that contains `tactical/` (the challenge repository's `data/`). The
command runs, in order and each with its own checks:

  1. pipeline/run.py         possession probabilities          -> <out>/ownership.csv
  2. pipeline/game_state.py  ball in play / stopped, per frame -> <out>/game_state.csv
  3. pipeline/fuse.py        H/A/D/X per frame                 -> <out>/<team>.csv
  4. the official Metrica validator on that exact file
  5. a copy of the technical document                   -> <out>/<team>.md

and prints the SHA-256 of what to upload. Re-running on the same data gives the same bytes.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def step(title, *args):
    print(f'\n== {title}', flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=os.environ.get('MTAH_DATA_DIR'),
                   help='folder containing tactical/ (default: $MTAH_DATA_DIR)')
    p.add_argument('--out', type=Path, default=ROOT / 'results')
    p.add_argument('--cache', type=Path, default=ROOT / 'work/cache')
    p.add_argument('--team', default='Inazuma', help='names the uploaded files: <team>.csv, <team>.md')
    p.add_argument('--no-tags', action='store_true', help='ignore Smart Tagging in both components')
    args = p.parse_args()
    if args.data_dir is None:
        p.error('--data-dir is required (or set MTAH_DATA_DIR)')
    data, out, cache = args.data_dir.resolve(), args.out.resolve(), args.cache.resolve()
    if not (data / 'tactical/BAR-ATH_tactical_atd.txt').exists():
        p.error(f'{data}/tactical/BAR-ATH_tactical_atd.txt not found: --data-dir must contain tactical/')
    tags = ['--no-tags'] if args.no_tags else []
    submission = out / f'{args.team}.csv'

    step('1/4 possession', 'pipeline/run.py', '--data-dir', data, '--out', out, '--cache', cache, *tags)
    step('2/4 game state', 'pipeline/game_state.py', '--data-dir', data, '--out', out, '--cache', cache, *tags)
    step('3/4 fusion', 'pipeline/fuse.py', '--ownership', out / 'ownership.csv', '--states', out / 'game_state.csv',
         '--state-metadata', out / 'game_state.meta.json', '--out', submission)
    step('4/4 official validator', 'third_party/metrica/evaluation/validate_submission.py', submission)

    document = out / f'{args.team}.md'
    shutil.copyfile(ROOT / 'TECHNICAL_DOCUMENT.md', document)
    print('\nUpload these two files:')
    for f in (submission, document):
        print(f'  {f}\n    sha256 {sha256(f)}')


if __name__ == '__main__':
    main()
