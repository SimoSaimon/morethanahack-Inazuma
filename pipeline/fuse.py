"""Combine ownership probabilities with game-state labels."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import argparse
import json
from pathlib import Path
import time
from possession.models import load
from possession.integration import read_states,read_ownership,fuse,write_submission
from possession.io import validate_submission,sha256

from possession.timeline import BLOCKS,N_FRAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ownership',required=True); p.add_argument('--states',required=True)
    p.add_argument('--state-metadata',required=True); p.add_argument('--out',required=True,type=Path)
    p.add_argument('--ownership-metadata',help='Defaults to run_manifest.json beside ownership.csv')
    p.add_argument('--model',default=str(ROOT/'models/possession_model.json'))
    args=p.parse_args(); start=time.perf_counter(); model=load(args.model)
    om_path=Path(args.ownership_metadata) if args.ownership_metadata else Path(args.ownership).parent/'run_manifest.json'
    om=json.loads(om_path.read_text())
    if om['ownership_sha256']!=sha256(args.ownership) or om['model_sha256']!=sha256(args.model):
        raise ValueError('Ownership file/model does not match its recorded producer')
    if om['timeline']!='BAR-ATH_tactical' or om['fps']!=25 or om['frame_origin']!=0 or om['rows']!=N_FRAMES:
        raise ValueError('Ownership metadata is not on the target tactical timeline')
    states,metadata=read_states(args.states,args.state_metadata,N_FRAMES)
    home,quality=read_ownership(args.ownership,N_FRAMES)
    labels,margin=fuse(home,quality,states,BLOCKS,model['decoder'])
    args.out.parent.mkdir(parents=True,exist_ok=True); write_submission(args.out,labels)
    report={'validation':validate_submission(args.out,N_FRAMES),'state_source_sha256':sha256(args.states),
            'ownership_source_sha256':sha256(args.ownership),'state_metadata':metadata,
            'state_policy':'stopped -> D; unknown -> X; in_play -> temporally decoded H/A',
            'decoder':model['decoder'],'runtime_s':time.perf_counter()-start}
    args.out.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
