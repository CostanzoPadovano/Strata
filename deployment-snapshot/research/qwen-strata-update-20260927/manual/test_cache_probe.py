import unittest
from cache_probe import padded_prompt,token_hash

class Planning(unittest.TestCase):
    def test_padding_preserves_roles_and_prefix(self):
        ids,turn=padded_prompt([8,4,5,9,8,3],13,8,9,[1,2,7])
        self.assertEqual(len(ids),13)
        self.assertEqual(turn,11)
        self.assertEqual(ids[:3],[8,4,5])
        self.assertEqual(ids[-3:],[9,8,3])
    def test_outside_finite_plan_rejected(self):
        for n in (0,4,40001):
            with self.assertRaises(ValueError): padded_prompt([8,4,5,9,8,3],n,8,9,[1])
    def test_hash_exact_order(self):
        self.assertEqual(token_hash([1,2,3]),token_hash([1,2,3]))
        self.assertNotEqual(token_hash([1,2,3]),token_hash([1,3,2]))

if __name__=='__main__': unittest.main()
