"""Game-state component: ball in play or stopped, per tactical frame -> game_state.csv for pipeline/fuse.py."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import argparse
import json
from pathlib import Path
import numpy as np
from possession.io import load_tracking,load_tags
from possession.dead_ball import dead_mask,set_piece_tags,speed_valley,P
from possession.timeline import BLOCKS,N_FRAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True,type=Path); p.add_argument('--out',required=True,type=Path)
    p.add_argument('--cache',default='work/cache'); p.add_argument('--no-tags',action='store_true')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    stem=str(args.data_dir/'tactical/BAR-ATH_tactical')
    raw,metadata,_=load_tracking(Path(stem+'_atd.xml'),Path(stem+'_atd.txt'),args.cache)
    if len(raw)!=N_FRAMES or metadata['fps']!=25: raise ValueError('Expected BAR-ATH tactical at 25 fps')
    # The XML labels the speed channel kph, but it matches m/s computed from the positions.
    raw=np.asarray(raw,dtype=np.float64)
    tags=[] if args.no_tags else set_piece_tags(load_tags(Path(stem+'_smart_tagging.xml'),N_FRAMES,25))
    # Speed thresholds from this match's own unlabelled player speeds (graded halves only).
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p90=np.concatenate([np.nanpercentile(raw[a:b,:-1,2],90,axis=1) for a,b in BLOCKS])
    enter,leave=speed_valley(p90); params=dict(P,slow_enter=enter,slow_exit=leave)
    dead=dead_mask(raw[:,-1],raw[:,:-1],p=params,tags=tags,blocks=BLOCKS)
    with (args.out/'game_state.csv').open('w',newline='',encoding='utf-8') as f:
        f.write('frame,state\n')
        f.writelines(f'{i},{"interrotta" if d else "in gioco"}\n' for i,d in enumerate(dead))
    (args.out/'game_state.meta.json').write_text(json.dumps(
        {'schema_version':1,'timeline':'BAR-ATH_tactical','fps':25,'frame_origin':0,'n_frames':N_FRAMES},indent=2)+'\n')
    print(json.dumps({'stopped_share':float(dead.mean()),'tags_used':len(tags),'slow_enter':enter,'slow_exit':leave}))


if __name__=='__main__': main()
