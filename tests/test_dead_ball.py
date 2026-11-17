import unittest
import numpy as np
from possession.dead_ball import runs,hysteresis,tag_restarts,set_piece_tags,half_kickoffs,P


class DeadBallTests(unittest.TestCase):
    def test_runs_and_hysteresis(self):
        self.assertEqual(runs(np.array([0,1,1,0,1],bool)),[(1,3),(4,5)])
        self.assertEqual(hysteresis(np.array([3,1.5,2.2,2.6,1.0]),2.0,2.5).tolist(),[0,1,1,0,1])
    def test_tag_restart_never_leaves_a_stoppage_under_min_dead(self):
        # a restart landing just after a stoppage starts, or before a later one
        for a,b in [(100,400),(100,150),(190,400)]:
            dead=np.zeros(1000,bool); dead[a:b]=True
            out=tag_restarts(dead,np.array([180,200]),[100],P)
            self.assertTrue(all(e-s>=P['min_dead'] for s,e in runs(out)),(a,b,runs(out)))
    def test_tag_opens_a_missed_stoppage_up_to_its_restart(self):
        out=tag_restarts(np.zeros(1000,bool),np.array([180,200]),[100],P)
        self.assertEqual(runs(out),[(100,180)])
    def test_set_piece_tags_skip_free_kicks_and_other_codes(self):
        ev=[{'code':'SET PIECES','start':10,'tags':{'Type':'Corner'}},
            {'code':'SET PIECES','start':20,'tags':{'Type':'Free Kick'}},
            {'code':'BUILD UP','start':30,'tags':{}}]
        self.assertEqual(set_piece_tags(ev),[(10,'Corner')])
    def test_half_kickoff_from_its_tag_snapped_to_a_kick(self):
        ball=np.full((3000,3),np.nan); kicks=np.array([500,1100,2590])
        tags=[(1000,'Kick Off (Start)'),(2000,'Throw-in'),(2500,'Kick Off (Start)')]
        # first block: the tag starts before it, the kick it marks is inside
        self.assertEqual(half_kickoffs(ball,kicks,tags,[(1080,1900),(2100,3000)],P),[1100,2590])
    def test_half_kickoff_without_tags_is_the_first_kick_from_the_centre_spot(self):
        ball=np.full((3000,3),np.nan); ball[2200]=[.9,.5,0]; ball[2400]=[.51,.5,0]
        self.assertEqual(half_kickoffs(ball,np.array([2200,2400]),[],[(2100,3000),(0,100)],P),[2400,None])


if __name__=='__main__': unittest.main()
