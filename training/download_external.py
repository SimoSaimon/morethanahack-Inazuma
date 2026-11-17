"""Download the fourteen public IDSSE files required to reproduce training (~2.6 GB)."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path); args=p.parse_args()
    manifest=json.loads((ROOT/'models/external_sources.json').read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    for source in manifest['files']:
        dest=args.out/source['file']
        if dest.exists():
            if dest.stat().st_size!=source['bytes'] or digest(dest)!=source['sha256']:
                raise ValueError('Existing file differs from the recorded source: '+str(dest))
            print('Verified existing '+dest.name,flush=True); continue
        temporary=dest.with_suffix(dest.suffix+'.part')
        try:
            with urllib.request.urlopen(source['url'],timeout=120) as response,temporary.open('wb') as f:
                while block:=response.read(1024*1024): f.write(block)
            if temporary.stat().st_size!=source['bytes'] or digest(temporary)!=source['sha256']:
                raise ValueError('Downloaded content differs from the frozen training source')
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)
        print('Downloaded and verified '+dest.name,flush=True)


if __name__=='__main__': main()
