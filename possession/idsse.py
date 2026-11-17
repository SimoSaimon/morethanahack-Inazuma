"""Read external tracking and ownership labels into separate arrays."""
from array import array
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from .io import sha256


def prepare(xml_path,meta_path,cache_dir):
    xml_path=Path(xml_path); meta_path=Path(meta_path); cache=Path(cache_dir)
    cache.mkdir(parents=True,exist_ok=True)
    digest=sha256(xml_path); mdigest=sha256(meta_path)
    dest=cache/(digest[:16]+'-'+mdigest[:8]+'-ownership-v1.npz')
    if dest.exists(): return dest
    root=ET.parse(meta_path).getroot()
    teams={t.attrib['TeamId']:0 if t.attrib['Role']=='home' else 1 for t in root.findall('.//Teams/Team')}
    if sorted(teams.values())!=[0,1]: raise ValueError('Need exactly two teams')
    halves={}; series=None; stack=[]; pitch=None
    for event,el in ET.iterparse(xml_path,events=('start','end')):
        if event=='start':
            stack.append(el)
            if el.tag=='FrameSet':
                section=el.attrib['GameSection']; team=el.attrib['TeamId']
                series={'id':el.attrib['PersonId'],'team':team,'n':array('i'),'x':array('f'),'y':array('f'),'status':array('b'),'possession':array('b')}
            continue
        if el.tag=='PitchSize': pitch=[float(el.attrib[k]) for k in ('X','Y')]
        if el.tag=='Frame':
            at=el.attrib; series['n'].append(int(at['N']))
            series['x'].append(float(at['X'])); series['y'].append(float(at['Y']))
            if series['team']=='BALL':
                series['status'].append(int(at['BallStatus']))
                series['possession'].append(int(at['BallPossession']))
            # Free children immediately; no multi-million-element tree in memory.
            stack[-2].remove(el)
        if el.tag=='FrameSet':
            halves.setdefault(section,[]).append(series); series=None
            stack[-2].remove(el)
        stack.pop()
    if set(halves)!={'firstHalf','secondHalf'} or pitch is None: raise ValueError('Incomplete match XML')
    output={}; summary=[]
    for half,section in enumerate(('firstHalf','secondHalf')):
        items=halves[section]; balls=[x for x in items if x['team']=='BALL']
        if len(balls)!=1: raise ValueError('Need one ball series per half')
        ball=balls[0]; frame=np.asarray(ball['n'])
        if not len(frame) or not np.all(np.diff(frame)==1): raise ValueError('Non-contiguous ball timeline')
        players=[x for x in items if x['team'] in teams]; n=len(frame)
        raw=np.full((n,len(players)+1,3),np.nan,np.float32)
        for j,s in enumerate(players+[ball]):
            indices=np.asarray(s['n'])-frame[0]
            if len(indices)!=len(np.unique(indices)) or np.any(indices<0) or np.any(indices>=n): raise ValueError('Invalid player timeline')
            raw[indices,j,0]=np.asarray(s['x'])/pitch[0]+.5
            raw[indices,j,1]=np.asarray(s['y'])/pitch[1]+.5
        status=np.asarray(ball['status'],np.int8)
        if not np.isin(status,[0,1]).all(): raise ValueError('Unknown match status')
        # Drop ambiguous post-half tails rather than learning their static positions.
        active=np.flatnonzero(status==1)
        if not len(active): raise ValueError('No active play')
        a=int(active[0]); b=int(active[-1])+1
        output[f'raw{half}']=raw[a:b]
        output[f'state{half}']=status[a:b].astype(np.int8)
        owner=np.asarray(ball['possession'],np.int8)[a:b]
        output[f'owner{half}']=np.where(np.isin(owner,[1,2]),owner-1,-1).astype(np.int8)
        output[f'team{half}']=np.array([teams[p['team']] for p in players],np.int8)
        summary.append({'section':section,'source_first_frame':int(frame[a]),'frames':b-a,'dead_fraction':float((1-status[a:b]).mean()),'slots':len(players)})
    info={'source_sha256':digest,'metadata_sha256':mdigest,'pitch_m':pitch,'fps':25,
          'match_id':root.find('.//General').attrib['MatchId'],'halves':summary}
    output['metadata']=np.array(json.dumps(info))
    np.savez(dest,**output)
    return dest

