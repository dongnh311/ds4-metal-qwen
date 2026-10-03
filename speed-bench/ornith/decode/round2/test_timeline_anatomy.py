"""Unit tests for timeline_anatomy (python3 -m unittest); synthetic timelines, no GPU."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import timeline_anatomy as ta  # noqa: E402

MS = 1_000_000  # ns


def _e(seq, idx, dur, gap, kernel, n=1):
    # start/end are not used by the anatomy (other clock than B); keep them plausible
    return "E %d %d %d %d %.2f %.2f 0x1 %d 1x1x1 32x1x1 %s" % (seq, idx, 1000 * idx, 1000 * idx + int(dur * 1000),
                                                            dur, gap, n, kernel)


LINES = [
    "# ds4 encoder timeline; slide=0x0 pid=1",
    "# B <seq> <n_encoders> <gpu_start_ns> <gpu_end_ns>",
    # prefill: one 60 ms buffer
    "B 1 1 %d %d" % (0, 60 * MS),
    _e(1, 0, 59000.0, 0.0, "kernel_prefill_gemm"),
    # cycle 1: verify buffer (3 encoders) then draft buffer (2 encoders)
    "B 2 3 %d %d" % (61 * MS, 79 * MS),
    _e(2, 0, 100.0, 0.0, "kernel_big"),
    _e(2, 1, 5.0, 2.0, "kernel_small_a"),
    _e(2, 2, 4.0, 3.0, "kernel_small_b"),
    "B 3 2 %d %d" % (80 * MS, 81 * MS),
    _e(3, 0, 8.0, 0.0, "kernel_qwen35_mtp_concat"),
    _e(3, 1, 50.0, 1.0, "kernel_big"),
    # cycle 2: same shape
    "B 4 3 %d %d" % (82 * MS, 100 * MS),
    _e(4, 0, 100.0, 0.0, "kernel_big"),
    _e(4, 1, 5.0, 2.0, "kernel_small_a"),
    _e(4, 2, 4.0, 3.0, "kernel_small_b"),
    "B 5 2 %d %d" % (101 * MS, 102 * MS),
    _e(5, 0, 8.0, 0.0, "kernel_qwen35_mtp_concat"),
    _e(5, 1, 50.0, 1.0, "kernel_big"),
]


class AnatomyTest(unittest.TestCase):
    def setUp(self):
        self.bufs = ta.parse(LINES)

    def test_parse_groups_encoders_under_buffers(self):
        self.assertEqual([b["seq"] for b in self.bufs], [1, 2, 3, 4, 5])
        self.assertEqual(len(self.bufs[1]["encs"]), 3)
        self.assertEqual(self.bufs[2]["encs"][0]["kernel"], "kernel_qwen35_mtp_concat")

    def test_decode_buffers_skip_prefill(self):
        dec = ta.decode_buffers(self.bufs, prefill_ms=50.0)
        self.assertEqual([b["seq"] for b in dec], [2, 3, 4, 5])

    def test_anatomy_per_cycle(self):
        a = ta.anatomy(ta.decode_buffers(self.bufs), small_us=10.0)
        self.assertEqual(a["cycles"], 2)
        self.assertAlmostEqual(a["buffer_gpu_us"], 19000.0)        # 18 ms verify + 1 ms draft
        self.assertAlmostEqual(a["busy_us"], 167.0)                 # 100+5+4+8+50
        self.assertAlmostEqual(a["intra_gap_us"], 6.0)              # 2+3+1
        # idle between consecutive decode buffers: 79->80, 81->82, 100->101 = 3 ms over 2 cycles
        self.assertAlmostEqual(a["inter_idle_us"], 1500.0)
        self.assertAlmostEqual(a["encoders"], 5.0)
        self.assertAlmostEqual(a["small_encoders"], 3.0)            # 5, 4 and 8 us
        self.assertAlmostEqual(a["small_busy_us"], 17.0)
        names = [k for k, _, _ in a["kernels"]]
        self.assertEqual(names[0], "kernel_big")                    # 150 us per cycle, largest first
        self.assertEqual(dict((k, c) for k, c, _ in a["kernels"])["kernel_small_a"], 1.0)

    def test_no_cycle_marker_raises(self):
        bufs = ta.parse(LINES[:9])                                  # prefill + one verify, no draft
        with self.assertRaises(ValueError):
            ta.anatomy(ta.decode_buffers(bufs))


if __name__ == "__main__":
    unittest.main()
