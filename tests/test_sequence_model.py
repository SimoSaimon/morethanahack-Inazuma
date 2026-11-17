import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from possession.models import _convolve,_sequence_score,load,score,sampled_score


def tiny_model(rng,inputs=4,channels=3,dilations=(1,2)):
    layer=lambda o,i,k:{'w':rng.normal(size=(o,i,k)).tolist(),'b':rng.normal(size=o).tolist()}
    blocks=[]
    for d in dilations:
        a=layer(channels,channels,3); b=layer(channels,channels,1)
        blocks.append({'dilation':d,'w3':a['w'],'b3':a['b'],'w1':b['w'],'b1':b['b']})
    member={'inp':layer(channels,inputs,1),'blocks':blocks,'out':layer(1,channels,1)}
    return {'schema_version':1,'kind':'tcn','feature_names':[f'f{i}' for i in range(inputs)],'swap_sign':[-1,-1,1,1],
            'mean':[0.]*inputs,'scale':[1.,2.,1.,.5],'members':[member,member],'temperature':1.,'prediction_stride':1}


class SequenceModelTests(unittest.TestCase):
    def test_convolution_matches_a_direct_loop(self):
        rng=np.random.default_rng(1); h=rng.normal(size=(40,3)); w=rng.normal(size=(2,3,3)); b=rng.normal(size=2)
        for dilation in (1,2,7,50):
            expected=np.zeros((40,2))+b
            for t in range(40):
                for j in range(3):
                    u=t+(j-1)*dilation
                    if 0<=u<40: expected[t]+=w[:,:,j]@h[u]
            np.testing.assert_allclose(_convolve(h,w,b,dilation),expected,atol=1e-12)

    def test_team_swap_complements_and_neutral_input_is_zero(self):
        rng=np.random.default_rng(2); m=tiny_model(rng); x=rng.normal(size=(60,4)).astype(np.float32)
        swapped=x*np.asarray(m['swap_sign'],np.float32)
        np.testing.assert_allclose(score(x,m),-score(swapped,m),atol=1e-12)
        neutral=x.copy(); neutral[:,:2]=0
        np.testing.assert_array_equal(score(neutral,m),np.zeros(60))

    def test_blocks_are_independent_sequences(self):
        rng=np.random.default_rng(3); m=tiny_model(rng); x=rng.normal(size=(90,4)).astype(np.float32)
        out=sampled_score(x,[(0,40),(50,90)],m)
        np.testing.assert_allclose(out[:40],score(x[:40],m),atol=1e-12)
        np.testing.assert_allclose(out[50:],score(x[50:],m),atol=1e-12)
        np.testing.assert_array_equal(out[40:50],np.zeros(10))

    def test_load_accepts_valid_and_rejects_malformed(self):
        m=tiny_model(np.random.default_rng(4))
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'m.json'; f.write_text(json.dumps(m)); self.assertEqual(load(f)['kind'],'tcn')
            bad=json.loads(json.dumps(m)); bad['members'][0]['out']['b']=[0.,0.]; f.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load(f)
            bad=json.loads(json.dumps(m)); bad['members'][0]['blocks'][0]['dilation']=0; f.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load(f)


if __name__=='__main__': unittest.main()
