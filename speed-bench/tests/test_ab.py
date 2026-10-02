import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v41"))
import ab  # noqa: E402


def row(tps, **kw):
    r = {"gen_steady_tps": tps, "step_ms": 1000.0 / tps, "gpu_busy_ms": 50.0,
         "pread_ms": 10.0, "readahead_ms": 5.0, "host_ms": 20.0,
         "decode_hit_rate": 0.8, "wired_steady_gib": 29.0, "contaminated": False}
    r.update(kw)
    return r


class AbTest(unittest.TestCase):
    def test_terms_include_lookahead(self):
        self.assertIn("la_issued_per_tok", ab.TERMS)
        self.assertIn("la_used_per_tok", ab.TERMS)

    def test_order_suffixes_and_ratio(self):
        calls = []

        def run(side, suffix):
            calls.append((side, suffix))
            return row(10.0 if side == "a" else 11.0)

        res = ab.run_ab(run, "q")
        self.assertEqual(calls, [("a", "-q-0a"), ("b", "-q-1b"), ("b", "-q-2b"), ("a", "-q-3a")])
        self.assertAlmostEqual(res["ratio"], 1.1)
        self.assertEqual(res["runs"]["a"], [10.0, 10.0])
        self.assertAlmostEqual(res["terms"]["b"]["gpu_busy_ms"], 50.0)
        self.assertFalse(res["contaminated"])

    def test_contamination_propagates(self):
        res = ab.run_ab(lambda side, suffix: row(10.0, contaminated=side == "b"), "c")
        self.assertTrue(res["contaminated"])

    def test_parse_env(self):
        self.assertEqual(ab.parse_env(["A=1", "B="]), {"A": "1", "B": ""})
        with self.assertRaises(SystemExit):
            ab.parse_env(["NOEQUALS"])


class AbReuseTest(unittest.TestCase):
    def test_env_mismatch_names_stale_keys(self):
        want = {"a": {"DS4_X": "1"}, "b": {}}
        # a reused B run recorded with A's switch set is stale
        self.assertEqual(ab.env_mismatch({"DS4_X": "1"}, want, "b"), ["DS4_X"])
        self.assertEqual(ab.env_mismatch({"DS4_X": "1", "DS4_OTHER": "2"}, want, "a"), [])
        self.assertEqual(ab.env_mismatch({}, want, "a"), ["DS4_X"])

    def test_summary_line_without_a_side(self):
        res = ab.run_ab(lambda side, suffix: row(10.0, gen_steady_tps=0.0 if side == "a" else 10.0),
                        "z")
        self.assertIsNone(res["ratio"])
        self.assertIn("ratio n/a", ab.summary_line(res))


class AbMainTest(unittest.TestCase):
    def _main(self, argv, fake):
        old_argv, old_run = sys.argv, ab.phase0.run_one
        sys.argv = ["ab.py", "--label", "x", "--model", "/m", "--prompts", "/p"] + argv
        ab.phase0.run_one = fake
        try:
            return ab.main()
        finally:
            sys.argv, ab.phase0.run_one = old_argv, old_run

    def test_per_side_bins_and_plain(self):
        calls = []

        def fake(bin_dir, model, prompts, out, spec, extra_env=None, tag_suffix="", base_env=None):
            calls.append((bin_dir, base_env))
            return row(10.0 if bin_dir == "/bins/a" else 12.0)

        with tempfile.TemporaryDirectory() as tmp:
            self._main(["--out", tmp, "--a-bin", "/bins/a", "--b-bin", "/bins/b", "--plain"], fake)
            with open(os.path.join(tmp, "ab-x.json")) as fp:
                res = json.load(fp)
        self.assertEqual([c[0] for c in calls], ["/bins/a", "/bins/b", "/bins/b", "/bins/a"])
        self.assertTrue(all(c[1] == {} for c in calls))
        self.assertAlmostEqual(res["ratio"], 1.2)
        self.assertEqual((res["a_bin"], res["b_bin"], res["plain"]), ("/bins/a", "/bins/b", True))

    def test_default_one_bin_with_profile_env(self):
        calls = []

        def fake(bin_dir, model, prompts, out, spec, extra_env=None, tag_suffix="", base_env=None):
            calls.append((bin_dir, base_env))
            return row(10.0)

        with tempfile.TemporaryDirectory() as tmp:
            self._main(["--out", tmp, "--bin", "/bins/one"], fake)
        self.assertEqual({c[0] for c in calls}, {"/bins/one"})
        self.assertTrue(all(c[1] is None for c in calls))

    def _write_done(self, out, bin_dir, ds4_env):
        spec = ("switch", 8192, None, 512, False)   # ab defaults with --a-cache auto
        path = os.path.join(out, ab.phase0.run_tag(spec, "-x-0a") + ".result.json")
        with open(path, "w") as fp:
            json.dump(dict(row(10.0), bin_dir=bin_dir, ds4_env=ds4_env), fp)

    def test_reuse_refuses_other_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_done(tmp, "/bins/old", {})
            with self.assertRaises(SystemExit) as cm:
                self._main(["--out", tmp, "--a-cache", "auto", "--a-bin", "/bins/a",
                            "--b-bin", "/bins/b", "--plain"], lambda *a, **k: None)
        self.assertIn("bin", str(cm.exception))

    def test_reuse_refuses_profile_mode_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_done(tmp, "/bins/a", dict(ab.phase0.ENV))
            with self.assertRaises(SystemExit) as cm:
                self._main(["--out", tmp, "--a-cache", "auto", "--a-bin", "/bins/a",
                            "--b-bin", "/bins/b", "--plain"], lambda *a, **k: None)
        self.assertIn("profile env", str(cm.exception))

    def test_profile_mismatch(self):
        full = dict(ab.phase0.ENV)
        self.assertFalse(ab.profile_mismatch({}, True))
        self.assertTrue(ab.profile_mismatch(full, True))
        self.assertFalse(ab.profile_mismatch(full, False))
        self.assertTrue(ab.profile_mismatch({}, False))


if __name__ == "__main__":
    unittest.main()
