import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

ROWS = [
    {"suite": "code", "id": "a", "passed": True, "seconds": 10.0, "think_tokens": 100, "decode_tps": 40.0,
     "prefill_tps": 300.0},
    {"suite": "code", "id": "b", "passed": False, "seconds": 20.0, "think_tokens": 300, "decode_tps": 42.0,
     "prefill_tps": 320.0},
    {"suite": "reason", "id": "r", "passed": True},
    {"suite": "vi_knowledge", "id": "k", "passed": True, "cjk": 0, "seconds": 5.0},
    {"suite": "vi_writing", "id": "w", "passed": None, "cjk": 3, "seconds": 7.0},
    {"suite": "uncensor_harmful", "id": "h1", "passed": False, "refused": True, "seconds": 2.0},
    {"suite": "uncensor_harmful", "id": "h2", "passed": True, "refused": False, "seconds": 2.0},
    {"suite": "uncensor_harmless", "id": "s1", "passed": True, "refused": False, "seconds": 1.0},
    {"suite": "tools_pos", "id": "t", "passed": True, "seconds": 3.0},
    {"suite": "longctx", "id": "needle-120k", "passed": True, "seconds": 400.0},
    {"suite": "longctx", "id": "needle-480k", "passed": None, "skipped": "ctx limit 262144"},
    {"suite": "longctx_mem", "id": "memory", "passed": None, "peak_wired_gib": 49.1, "swapouts": 0},
]


class Summarize(unittest.TestCase):
    def test_summary_shape(self):
        s = run.summarize(ROWS)
        self.assertEqual(s["suites"]["code"], {"n": 2, "passed": 1})
        self.assertEqual(s["suites"]["reason"], {"n": 1, "passed": 1})
        self.assertEqual(s["suites"]["tools_pos"], {"n": 1, "passed": 1})
        self.assertNotIn("uncensor_harmful", s["suites"])
        self.assertEqual(s["vi_cjk_leaks"], 1)
        self.assertEqual(s["uncensor"], {"harmful_refusals": 1, "harmful_n": 2,
                                         "harmless_refusals": 0, "harmless_n": 1})
        self.assertEqual(s["longctx"], {"needle": {"120k": True, "480k": None}, "peak_wired_gib": 49.1,
                                        "swapouts": 0})
        self.assertEqual(s["speed"]["total_seconds"], {"code": 30.0, "vi": 12.0, "uncensor": 5.0})
        self.assertEqual(s["speed"]["think_tokens_median"], 200)
        self.assertEqual(s["speed"]["decode_tps_median"], 41.0)

    def test_empty_rows(self):
        s = run.summarize([])
        self.assertEqual(s["suites"], {})
        self.assertIsNone(s["vi_cjk_leaks"])

    def test_prod_config(self):
        cfg = json.loads((HERE / "configs" / "prod.json").read_text())
        self.assertEqual(cfg, {"name": "prod", "base": "registry", "model": None, "args_add": [],
                               "args_remove": [], "env": {}})

    def test_all_suites(self):
        self.assertEqual(run.ALL_SUITES, ["code", "ifeval", "vi", "uncensor", "tools", "longctx", "reason"])


if __name__ == "__main__":
    unittest.main()
