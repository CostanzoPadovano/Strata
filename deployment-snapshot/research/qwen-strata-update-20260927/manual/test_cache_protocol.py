import unittest
from cache_protocol import parse_cache_done

class Protocol(unittest.TestCase):
    def test_reuse_and_real_prefill(self):
        row=parse_cache_done('DONE 50 20001 2000 1000 stop 20 40 19000')
        self.assertEqual(row['prompt_reused'],19000)
        self.assertEqual(row['prefill_processed_tokens'],1000)
        self.assertEqual(row['draft_accepted'],20)
    def test_final_prompt_token_is_not_prefill(self):
        self.assertEqual(parse_cache_done('DONE 1 20001 2 25 length 0 0 20000')['prefill_processed_tokens'],0)
    def test_invalid_counters_fail_closed(self):
        for line in ('DONE 1 100 1 2 stop','DONE 1 100 1 2 stop 0 0 100',
                     'DONE 1 100 nan 2 stop 0 0 0','DONE 1 100 1 2 stop 4 3 0',
                     'DONE 1 100 1 2 stop 0 0 -1','DONE 1 100 1 -2 stop 0 0 0'):
            with self.subTest(line=line),self.assertRaises(ValueError): parse_cache_done(line)

if __name__=='__main__': unittest.main()
