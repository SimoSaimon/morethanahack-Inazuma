import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from possession.features import features
from possession.models import load,score,raw_score,sampled_score,sigmoid,canonical
from possession.temporal import decode
from possession.integration import read_states,read_ownership,fuse,write_submission
from possession.augmentation import degrade
from possession.io import validate_submission

MODEL=Path(__file__).parent.parent/'models/possession_model.json'


def raw_fixture(n=300):
    rng=np.random.default_rng(3); raw=np.full((n,23,3),np.nan,np.float32)
    raw[:,:22,:2]=rng.uniform(.15,.8,(22,2))
    raw[:,:22,0]+=np.arange(n)[:,None]/25*.7/105.3
    raw[:,-1,0]=.2+np.arange(n)/25*1.5/105.3; raw[:,-1,1]=.4
    return raw,np.array([0]*11+[1]*11)


class FeatureTests(unittest.TestCase):
    def test_signed_zero_serialization_is_canonical(self):
        self.assertEqual(json.dumps(canonical({'x':[-0.,0.,-.001]})), '{"x": [0.0, 0.0, -0.001]}')
    def test_team_swap_feature_and_prediction_equivariance(self):
        raw,t=raw_fixture(); x,s,_=features(raw,t); y,_,_=features(raw,1-t)
        np.testing.assert_allclose(y,x*np.asarray(s['swap_sign']),atol=1e-6)
        model=load(MODEL)
        np.testing.assert_allclose(sigmoid(score(x,model)),1-sigmoid(score(y,model)),atol=1e-12)

    def test_player_slot_permutation(self):
        raw,t=raw_fixture(); order=np.random.default_rng(8).permutation(22)
        swapped=np.concatenate([raw[:,order,:],raw[:,-1:,:]],axis=1)
        a,_,_=features(raw,t); b,_,_=features(swapped,t[order])
        np.testing.assert_allclose(a,b,atol=5e-6)
        # a network is continuous, so it follows the float32 noise of the features (logits span several units)
        np.testing.assert_allclose(score(a,load(MODEL)),score(b,load(MODEL)),atol=1e-4)

    def test_field_reflection(self):
        raw,t=raw_fixture(); reflected=raw.copy(); reflected[:,:,:2]=1-reflected[:,:,:2]
        a,_,_=features(raw,t); b,_,_=features(reflected,t)
        np.testing.assert_allclose(a,b,atol=2e-5)
        np.testing.assert_allclose(score(a,load(MODEL)),score(b,load(MODEL)),atol=1e-4)

    def test_no_observations_are_finite_and_neutral(self):
        raw,t=raw_fixture(); raw[:]=np.nan; x,_,d=features(raw,t)
        self.assertTrue(np.isfinite(x).all()); self.assertTrue(np.all(d['quality']==0))
        np.testing.assert_array_equal(score(x,load(MODEL)),np.zeros(len(x)))

    def test_gap_does_not_create_ball_observations(self):
        raw,t=raw_fixture(); raw[50:150,-1,:]=np.nan
        x,s,d=features(raw,t); self.assertFalse(d['ball_seen'][50:150].any())
        self.assertEqual(x[80,s['feature_names'].index('ball_seen')],0)
        # Only explicit memory may carry earlier evidence; physical geometry is absent.
        self.assertEqual(x[80,s['feature_names'].index('geometry')],0)

    def test_speed_channel_is_never_an_input(self):
        raw,t=raw_fixture(); a,_,_=features(raw,t); raw[:,:,2]=99999; b,_,_=features(raw,t)
        np.testing.assert_array_equal(a,b)

    def test_chunk_size_does_not_change_result(self):
        raw,t=raw_fixture(); a,_,_=features(raw,t,chunk=60); b,_,_=features(raw,t,chunk=1000)
        np.testing.assert_allclose(a,b,atol=1e-6)

    def test_stress_is_deterministic_and_preserves_source(self):
        raw,t=raw_fixture(1000); before=raw.copy()
        a=degrade(raw,t,[105.3,68],'test',0,'occluded'); b=degrade(raw,t,[105.3,68],'test',0,'occluded')
        np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(raw,before)

    def test_half_features_have_no_memory_of_other_half(self):
        raw,t=raw_fixture(); empty=np.full_like(raw,np.nan)
        x,s,_=features(empty,t)
        for key in ['memory_past_3s','memory_past_12s','memory_future_3s','memory_future_12s']:
            self.assertTrue(np.all(x[:,s['feature_names'].index(key)]==0))


class TemporalTests(unittest.TestCase):
    def test_variable_cost_and_margin_match_brute_force(self):
        scores=np.array([-.4,1.2,.8,-2,.7]); quality=np.array([1.,0,.6,1,0]); cost=.3; gap=2
        p,m=decode(scores,quality,np.ones(5),[(0,5)],fps=1,switch_cost=cost,gap_penalty=gap,with_margin=True)
        paths=list(itertools.product((0,1),repeat=5)); objective=[]
        for path in paths:
            total=sum(-np.logaddexp(0,scores[i] if v else -scores[i]) for i,v in enumerate(path))
            total-=sum(cost*(1+gap*(1-quality[i])) for i in range(1,5) if path[i]!=path[i-1])
            objective.append(total)
        best=paths[int(np.argmax(objective))]; np.testing.assert_array_equal(p,best)
        for t in range(5):
            alt=max(o for path,o in zip(paths,objective) if path[t]!=best[t])
            self.assertAlmostEqual(m[t],max(objective)-alt)

    def test_stoppage_overrides_strong_ownership(self):
        p,_=fuse(np.full(30,.999999),np.ones(30),np.zeros(30),[(0,30)],{'switch_cost':.2,'gap_penalty':0})
        self.assertTrue(np.all(p==2))

    def test_unknown_is_x_and_never_assumed_live(self):
        p,_=fuse(np.full(30,.99),np.ones(30),np.full(30,-1),[(0,30)],{'switch_cost':.2,'gap_penalty':0})
        self.assertTrue(np.all(p==3))

    def test_half_and_dead_reset(self):
        s=np.r_[np.full(10,10.),np.full(10,-10.)]; states=np.ones(20); states[8:12]=0
        p,_=decode(s,np.ones(20),states,[(0,20)],switch_cost=100)
        self.assertTrue(np.all(p[:8]==0)); self.assertTrue(np.all(p[8:12]==2)); self.assertTrue(np.all(p[12:]==1))
        p,_=decode(s,np.ones(20),np.ones(20),[(0,8),(12,20)],switch_cost=100)
        self.assertTrue(np.all(p[8:12]==3)); self.assertTrue(np.all(p[12:]==1))

    def test_invalid_inputs_fail(self):
        for prob in [np.array([np.nan]),np.array([1.1])]:
            with self.assertRaises(ValueError):fuse(prob,np.ones(1),np.ones(1),[(0,1)],{'switch_cost':.2,'gap_penalty':0})
        with self.assertRaises(ValueError):decode(np.zeros(2),np.ones(2),np.ones(2),[(0,2),(1,2)])
        with self.assertRaises(ValueError):decode(np.zeros(2),np.ones(2),np.ones(2),[(0,2)],switch_cost=-1)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.meta={'schema_version':1,'timeline':'BAR-ATH_tactical','fps':25,'frame_origin':0,'n_frames':3}
        self.path=self.root/'states.csv'; self.mp=self.root/'states.meta.json'; self.mp.write_text(json.dumps(self.meta))
    def tearDown(self):self.tmp.cleanup()
    def test_italian_states_and_csv_roundtrip(self):
        self.path.write_text('frame,state\n0,in gioco\n1,interrotta\n2,incerta\n')
        s,_=read_states(self.path,self.mp,3); np.testing.assert_array_equal(s,[1,0,-1])
        p,_=fuse([.9,.9,.9],[1,1,1],s,[(0,3)],{'switch_cost':.2,'gap_penalty':0})
        out=self.root/'out.csv'; write_submission(out,p); self.assertTrue(validate_submission(out,3)['valid'])
        self.assertEqual(out.read_text(),'frame,label\n0,H\n1,D\n2,X\n')
    def test_wrong_timeline_rejected(self):
        self.meta['timeline']='BAR-ATH_broadcast'; self.mp.write_text(json.dumps(self.meta))
        with self.assertRaises(ValueError):read_states(self.path,self.mp,3)
    def test_duplicate_missing_and_extra_rows_rejected(self):
        for text in ['frame,state\n0,in_play\n0,stopped\n2,unknown\n','frame,state\n0,in_play\n',
                     'frame,state\n0,in_play,extra\n1,stopped\n2,unknown\n']:
            self.path.write_text(text)
            with self.assertRaises(ValueError):read_states(self.path,self.mp,3)
    def test_ferma_is_ambiguous_and_rejected(self):
        self.path.write_text('frame,state\n0,ferma\n1,stopped\n2,unknown\n')
        with self.assertRaises(ValueError):read_states(self.path,self.mp,3)
    def test_wrong_fps_rejected(self):
        self.meta['fps']=30; self.mp.write_text(json.dumps(self.meta))
        with self.assertRaises(ValueError):read_states(self.path,self.mp,3)


if __name__=='__main__':unittest.main()
