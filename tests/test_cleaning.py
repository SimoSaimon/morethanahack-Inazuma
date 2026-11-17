import unittest
import numpy as np
from possession.cleaning import fix_goalkeeper_team


def fixture(n=400):
    raw = np.full((n, 25, 3), np.nan, np.float32); teams = np.array([0] * 12 + [1] * 12)
    raw[:, :10, 0] = np.linspace(.2, .6, 10); raw[:, 10, 0] = .04          # team 0 defends x=0; slot 11 keeper
    raw[:, 12:22, 0] = np.linspace(.4, .8, 10); raw[:, 22, 0] = .96        # team 1 defends x=1; slot 23 keeper
    raw[:, :, 1] = .5; raw[:, 11, :] = np.nan; raw[:, 23, :] = np.nan      # one empty slot per team
    return raw, teams


class GoalkeeperTests(unittest.TestCase):
    def test_clean_tracking_is_untouched(self):
        raw, teams = fixture(); out, report = fix_goalkeeper_team(raw, teams)
        np.testing.assert_array_equal(out, raw); self.assertEqual(report, [])

    def test_keeper_in_the_other_goal_changes_team(self):
        raw, teams = fixture(); raw[100:200, 22, 0] = .06
        out, report = fix_goalkeeper_team(raw, teams)
        self.assertEqual(report, [{'slot': 23, 'to_slot': 12, 'start': 100, 'end': 200}])
        self.assertTrue(np.isnan(out[100:200, 22]).all())
        np.testing.assert_array_equal(out[100:200, 11], raw[100:200, 22])
        np.testing.assert_array_equal(out[:100], raw[:100])

    def test_attacker_in_the_box_is_not_a_keeper(self):
        raw, teams = fixture(); raw[100:200, 21, 0] = .05
        out, report = fix_goalkeeper_team(raw, teams)
        self.assertEqual(report, [])


if __name__ == '__main__': unittest.main()
