"""No-GPU regressions for rejection of incomplete/misaligned wait evidence."""
import copy
import unittest
from verify_wait_profile import validate_frames,rank_decode_cpu_wait


def fixture():
    names=['arithmetic']+[f'decode_screen_{i}' for i in range(3)]
    rows=[{'layer':layer,'windows':3,'calls':[6,6,6],'waited':[0,0,1],
           'blocked_ns':[0,0,(layer+1)*(index+1)*1000]}
          for index,_ in enumerate(names) for layer in range(48)]
    return rows,names


class WaitProfiles(unittest.TestCase):
    def test_complete_ranked_decode_frames(self):
        rows,names=fixture()
        frames=validate_frames(rows,names)
        rank=rank_decode_cpu_wait(frames)
        self.assertEqual(rank['completed_windows'],9)
        self.assertEqual([row['layer'] for row in rank['ranking']],list(reversed(range(48))))
        self.assertAlmostEqual(rank['ranking'][0]['cpu_blocked_ms'],0.432)
        self.assertAlmostEqual(rank['cpu_blocked_ms_total'],10.584)

    def test_incomplete_or_duplicate_frames_are_rejected(self):
        rows,names=fixture()
        for bad_rows,bad_names in ((rows[:-1],names),(rows,names[:-1]),(rows,[names[0]]*4)):
            with self.assertRaises(ValueError):validate_frames(bad_rows,bad_names)
        self.assertEqual(validate_frames([],[]),[])
        with self.assertRaises(ValueError):validate_frames([],['a']*17)
        with self.assertRaises(ValueError):rank_decode_cpu_wait(validate_frames(rows[:48],names[:1]))

    def test_invalid_counters_and_layer_mapping_are_rejected(self):
        rows,names=fixture()
        for key,value in (('layer',1),('layer',False),('windows',0),('windows',True),('calls',[6,6,5]),
                          ('calls',[7,7,7]),('calls',[0,0,0]),('waited',[0,0,7]),
                          ('blocked_ns',[1,0,1000]),('blocked_ns',[0,0,-1]),
                          ('blocked_ns',[0,0,2**64]),('waited',[0,0,False]),('waited',[0,0])):
            bad=copy.deepcopy(rows)
            bad[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_frames(bad,names)


if __name__=='__main__':unittest.main(verbosity=2)
