"""Strict parsers. No network calls and no silent padding/trimming of source data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def metadata(path):
    root=ET.parse(path).getroot()
    teams={x.attrib['id']:x.findtext('Name') for x in root.findall('./Metadata/Teams/Team')}
    score=root.find('./Metadata/Sessions/Session/MatchParameters/Score')
    home=score.attrib['idLocalTeam']; away=score.attrib['idVisitingTeam']
    if home==away or set(teams)!={home,away}:
        raise ValueError('Expected two distinct teams and explicit home/away metadata')
    players={p.attrib['id']:p.attrib['teamId'] for p in root.findall('./Metadata/Players/Player')}
    channels={p.attrib['id']:(p.attrib['playerId'],p.attrib['channelId']) for p in root.findall('./Metadata/PlayerChannels/PlayerChannel')}
    fmt=root.find('./DataFormatSpecifications/DataFormatSpecification')
    groups=fmt.find("SplitRegister[@separator=';']").findall('SplitRegister')
    ids=[]
    for group in groups:
        refs=[channels[c.attrib['playerChannelId']] for c in group.findall('PlayerChannelRef')]
        if len(refs)!=3 or len({p for p,c in refs})!=1 or [c for p,c in refs]!=['x','y','s']:
            raise ValueError('Unsupported channel ordering; expected x,y,s per track')
        ids.append(refs[0][0])
    ball=[x.attrib['channelId'] for x in fmt.findall("SplitRegister[@separator=',']/BallChannelRef")]
    if ball!=['x','y','s']:
        raise ValueError('Unsupported ball channel order')
    if len(set(ids))!=len(ids) or any(players[p] not in teams for p in ids):
        raise ValueError('Invalid track identifiers/team mapping')
    return {'fps':float(root.findtext('./Metadata/GlobalConfig/FrameRate')),
            'team_names':[teams[home],teams[away]], 'track_ids':ids,
            'team_of_track':[0 if players[p]==home else 1 for p in ids],
            'n_frames':int(fmt.attrib['endFrame'])-int(fmt.attrib['startFrame'])+1,
            'start_frame':int(fmt.attrib['startFrame']),
            'speed_unit_xml':root.findtext(".//Channel[@id='s']/Unit"),
            'pitch_xml':[float(root.findtext('./Metadata/Sessions/Session/FieldSize/'+k)) for k in ('Width','Height')]}


def load_tracking(meta_path, raw_path, cache_dir):
    meta=metadata(meta_path)
    if meta['start_frame']!=1:
        raise ValueError('This loader requires a one-based source frame counter')
    raw_hash=sha256(raw_path); meta_hash=sha256(meta_path)
    key=hashlib.sha256((raw_hash+meta_hash+'parser-v1').encode()).hexdigest()
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    dest=cache_dir/(key+'.npy'); marker=cache_dir/(key+'.json')
    shape=(meta['n_frames'],len(meta['track_ids'])+1,3)
    hit=False
    if dest.exists() and marker.exists():
        arr=np.load(dest,mmap_mode='r',allow_pickle=False)
        if arr.shape!=shape or arr.dtype!=np.float32:
            raise ValueError('Cache structure does not match source; remove the cache and rerun')
        hit=True
    else:
        tmp=dest.with_suffix('.partial.npy')
        arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=shape)
        count=0
        try:
            with open(raw_path,encoding='utf-8') as f:
                for i,line in enumerate(f):
                    parts=line.strip().split(':')
                    if len(parts)!=3 or int(parts[0])!=i+1 or i>=shape[0]:
                        raise ValueError(f'Invalid or nonsequential frame at line {i+1}')
                    groups=parts[1].split(';')+[parts[2]]
                    if len(groups)!=shape[1] or any(g.count(',')!=2 for g in groups):
                        raise ValueError(f'Invalid object count at frame {i}')
                    # fromstring in NumPy 2 rejects trailing malformed tokens.
                    values=np.fromstring(','.join(groups),sep=',',dtype=np.float32)
                    if values.size!=shape[1]*3 or np.isinf(values).any():
                        raise ValueError(f'Invalid numeric values at frame {i}')
                    arr[i]=values.reshape(shape[1],3); count+=1
            if count!=shape[0]:
                raise ValueError(f'Truncated tracking: {count} != {shape[0]}')
            arr.flush(); del arr
            arr=None
            tmp.replace(dest)
            marker.write_text(json.dumps({'raw_sha256':raw_hash,'metadata_sha256':meta_hash,'shape':shape},sort_keys=True))
        except Exception:
            if arr is not None: del arr
            tmp.unlink(missing_ok=True)
            raise
        arr=np.load(dest,mmap_mode='r',allow_pickle=False)
    return arr,meta,{'raw_sha256':raw_hash,'metadata_sha256':meta_hash,'cache_hit':hit,'cache_bytes':dest.stat().st_size}


def load_homography(path,n_frames):
    expected='frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid'
    with open(path,encoding='utf-8-sig') as f:
        if f.readline().strip()!=expected:
            raise ValueError('Unexpected homography schema')
    h=np.loadtxt(path,delimiter=',',skiprows=1,dtype=np.float64,ndmin=2)
    if h.shape!=(n_frames,12) or not np.array_equal(h[:,0],np.arange(n_frames)):
        raise ValueError('Homography rows must match zero-based video frames exactly')
    if not np.isin(h[:,-1],[0,1]).all():
        raise ValueError('Invalid homography validity flag')
    valid=h[:,-1]==1
    if not np.isfinite(h[valid,1:11]).all():
        raise ValueError('Nonfinite valid homography')
    return h[:,1:10].reshape(-1,3,3),h[:,-2],valid


def load_tags(path,n_frames,fps):
    if path is None:
        return []
    events=[]
    for el in ET.parse(path).iter('instance'):
        start=float(el.findtext('start')); end=float(el.findtext('end'))
        a=int(round(start*fps)); b=int(round(end*fps))
        if not np.isfinite([start,end]).all() or not (0<=a<b<=n_frames):
            raise ValueError('Tag interval outside the source timeline')
        events.append({'code':el.findtext('code').strip().upper(),'start':a,'stop':b,
                       'tags':{x.findtext('group'):x.findtext('text') for x in el.findall('label')}})
    return events


def validate_submission(path,n_frames):
    with open(path,encoding='utf-8',newline='') as f:
        if f.readline().strip()!='frame,label':
            raise ValueError('CSV header must be frame,label')
        count=0
        for i,line in enumerate(f):
            fields=line.strip().split(',')
            if len(fields)!=2 or fields[0]!=str(i) or fields[1] not in ('H','A','D','X'):
                raise ValueError(f'Invalid submission row {i}')
            count+=1
    if count!=n_frames:
        raise ValueError(f'Expected {n_frames} rows, got {count}')
    return {'valid':True,'rows':count,'sha256':sha256(path)}
