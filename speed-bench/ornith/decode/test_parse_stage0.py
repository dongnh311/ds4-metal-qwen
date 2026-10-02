"""Unit tests for parse_stage0.py (python3 -m unittest, no model)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_stage0 as p

LOG = """ds4: cb command batch: driver 30 us, queue-wait 5 us, gpu 812000 us, gap-from-prev-gpu-end 0 us
ds4: cb command batch: driver 20 us, queue-wait 4 us, gpu 9000 us, gap-from-prev-gpu-end 1000 us
ds4: cb command batch: driver 20 us, queue-wait 4 us, gpu 11000 us, gap-from-prev-gpu-end 3000 us
ds4: Ornith decode split ms/step (avg over 64 steps, pos=10..73): gdn-mix 4.00 gdn-moe 8.00 attn-mix 2.00 attn-moe 3.00 head 2.00
ds4: Ornith decode split ms/step (avg over 64 steps, pos=74..137): gdn-mix 6.00 gdn-moe 8.00 attn-mix 2.00 attn-moe 3.00 head 2.00
ds4: Ornith spec stats: 64 cycles (1 plain), accept 0.870, 1.850 tokens/cycle, ms/cycle target 14.00 draft 2.00 host 0.50 outside 0.20, ms/token 9.30
ds4: Ornith verify expert overlap: 3.10 of 8 per layer over 2048 layer-verifies
ds4: Ornith prefill: 1980.35 t/s, generation: 45.11 t/s
"""


class ParseTest(unittest.TestCase):
    def test_gen_tps(self):
        self.assertEqual(p.gen_tps(LOG), 45.11)

    def test_cb_decode_drops_prefill_buffers(self):
        d = p.cb_decode(LOG)        # the 812 ms prefill buffer is not decode
        self.assertEqual(d["n"], 2)
        self.assertAlmostEqual(d["gpu_ms"], 20.0)
        self.assertAlmostEqual(d["gap_ms"], 4.0)
        self.assertAlmostEqual(d["busy"], 20.0 / 24.0)

    def test_split_means(self):
        s = p.split_means(LOG, "decode split ms/step")
        self.assertAlmostEqual(s["gdn-mix"], 5.0)
        self.assertAlmostEqual(s["head"], 2.0)

    def test_last_line(self):
        self.assertIn("accept 0.870", p.last_line(LOG, "Ornith spec stats:"))
        self.assertIn("3.10 of 8", p.last_line(LOG, "Ornith verify expert overlap:"))

    def test_empty_log(self):
        self.assertIsNone(p.gen_tps(""))
        self.assertEqual(p.cb_decode("")["n"], 0)
        self.assertEqual(p.split_means("", "decode split ms/step"), {})
        self.assertIsNone(p.last_line("", "Ornith spec stats:"))


if __name__ == "__main__":
    unittest.main()
