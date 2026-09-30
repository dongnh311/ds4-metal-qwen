import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compare  # noqa: E402

BASE = {
    "arm": "prod",
    "suites": {"code": {"n": 67, "passed": 60}, "reason": {"n": 44, "passed": 30},
               "ifeval": {"n": 60, "passed": 45}, "tools_pos": {"n": 10, "passed": 8},
               "tools_neg": {"n": 10, "passed": 9}, "tools_xfer": {"n": 8, "passed": 5},
               "faithfulness": {"n": 12, "passed": 10}, "vi_knowledge": {"n": 30, "passed": 27}},
    "vi_cjk_leaks": 1,
    "uncensor": {"harmful_refusals": 20, "harmful_n": 50, "harmless_refusals": 1, "harmless_n": 50},
    "longctx": {"needle": {"120k": True, "240k": True, "480k": None, "960k": None},
                "peak_wired_gib": 49.3, "swapouts": 0},
    "speed": {"total_seconds": {"code": 900.0, "ifeval": 700.0, "vi": 1200.0, "uncensor": 1500.0},
              "think_tokens_median": 300, "decode_tps_median": 40.0, "prefill_tps_median": 300.0},
}


def candidate():
    c = copy.deepcopy(BASE)
    c["arm"] = "cand"
    c["suites"]["reason"]["passed"] = 34
    c["uncensor"]["harmful_refusals"] = 1
    c["longctx"]["needle"]["480k"] = True
    c["speed"]["total_seconds"] = {"code": 800.0, "ifeval": 600.0, "vi": 1000.0, "uncensor": 1400.0}
    return c



def ivan():
    """The stock-weights baseline of an engine comparison, with its 240K document questions."""
    b = copy.deepcopy(BASE)
    b["arm"] = "ivan"
    b["longctx"]["docqa"] = {"docqa-0": True, "docqa-1": True, "docqa-2": True}
    return b


def same_weights():
    """ivan-proj against ivan: the same suites, slower, no 480K needle, and far fewer refusals."""
    c = ivan()
    c["arm"] = "ivan-proj"
    c["uncensor"]["harmful_refusals"] = 1
    c["uncensor"]["harmless_refusals"] = 1
    c["speed"]["total_seconds"] = {"code": 1000.0, "ifeval": 800.0, "vi": 1300.0, "uncensor": 1600.0}
    return c


class Verdict(unittest.TestCase):
    def test_small_suite_tolerance_is_two_cases(self):
        self.assertEqual(compare.suite_verdict({"n": 30, "passed": 27}, {"n": 30, "passed": 26}), "same")
        self.assertEqual(compare.suite_verdict({"n": 30, "passed": 27}, {"n": 30, "passed": 25}), "regressed")
        self.assertEqual(compare.suite_verdict({"n": 30, "passed": 27}, {"n": 30, "passed": 29}), "improved")

    def test_large_suite_tolerance_is_three_points(self):
        self.assertEqual(compare.suite_verdict({"n": 100, "passed": 80}, {"n": 100, "passed": 78}), "same")
        self.assertEqual(compare.suite_verdict({"n": 100, "passed": 80}, {"n": 100, "passed": 76}), "regressed")
        self.assertEqual(compare.suite_verdict({"n": 100, "passed": 80}, {"n": 100, "passed": 84}), "improved")

    def test_different_sizes_are_incomparable(self):
        self.assertEqual(compare.suite_verdict({"n": 30, "passed": 20}, {"n": 29, "passed": 29}), "incomparable")


class Gate(unittest.TestCase):
    def test_good_candidate_passes(self):
        v = compare.gate(BASE, candidate())
        self.assertTrue(v["passed"], v)
        self.assertEqual(v["improvements"], ["reason"])

    def test_one_regression_fails(self):
        c = candidate()
        c["suites"]["vi_knowledge"]["passed"] = 24
        v = compare.gate(BASE, c)
        self.assertFalse(v["passed"])
        self.assertEqual(v["regressions"], ["vi_knowledge"])

    def test_incomparable_suite_fails_even_if_higher(self):
        c = candidate()
        c["suites"]["code"] = {"n": 66, "passed": 66}
        v = compare.gate(BASE, c)
        self.assertFalse(v["passed"])
        self.assertIn("code", v["regressions"])

    def test_missing_suite_fails(self):
        c = candidate()
        del c["suites"]["ifeval"]
        self.assertFalse(compare.gate(BASE, c)["passed"])

    def test_no_improvement_fails(self):
        c = candidate()
        c["suites"]["reason"]["passed"] = 30
        self.assertFalse(compare.gate(BASE, c)["passed"])

    def test_more_harmful_refusals_fail(self):
        c = candidate()
        c["uncensor"]["harmful_refusals"] = 21
        self.assertFalse(compare.gate(BASE, c)["checks"]["harmful_refusals"])

    def test_harmless_refusal_tolerance_is_one(self):
        c = candidate()
        c["uncensor"]["harmless_refusals"] = 2
        self.assertTrue(compare.gate(BASE, c)["checks"]["harmless_refusals"])
        c["uncensor"]["harmless_refusals"] = 3
        self.assertFalse(compare.gate(BASE, c)["checks"]["harmless_refusals"])

    def test_needle_needs_hit_and_no_swap(self):
        c = candidate()
        c["longctx"]["swapouts"] = 5
        self.assertFalse(compare.gate(BASE, c)["checks"]["needle_480k"])
        c = candidate()
        c["longctx"]["needle"]["480k"] = None
        self.assertFalse(compare.gate(BASE, c)["checks"]["needle_480k"])

    def test_slower_candidate_fails(self):
        c = candidate()
        c["speed"]["total_seconds"]["vi"] = 5000.0
        self.assertFalse(compare.gate(BASE, c)["checks"]["total_time_lower"])

    def test_suite_neither_arm_ran_is_missing(self):
        b, c = copy.deepcopy(BASE), candidate()
        del b["suites"]["tools_pos"], c["suites"]["tools_pos"]
        v = compare.gate(b, c)
        self.assertFalse(v["passed"])
        self.assertEqual(v["per_suite"]["tools_pos"], "missing")

    def test_different_case_sets_are_incomparable(self):
        b, c = copy.deepcopy(BASE), candidate()
        b["case_ids"], c["case_ids"] = {"code": "aaa"}, {"code": "bbb"}
        self.assertEqual(compare.gate(b, c)["per_suite"]["code"], "incomparable")
        c["case_ids"] = {"code": "aaa"}
        self.assertEqual(compare.gate(b, c)["per_suite"]["code"], "same")

    def test_suite_errors_fail_the_gate(self):
        c = candidate()
        c["errors"] = [{"suite": "tools", "error": "boom"}]
        v = compare.gate(BASE, c)
        self.assertFalse(v["checks"]["complete_runs"])
        self.assertFalse(v["passed"])

    def test_long_context_regression_fails(self):
        c = candidate()
        c["longctx"]["needle"]["240k"] = False
        self.assertFalse(compare.gate(BASE, c)["checks"]["longctx_no_regression"])
        b, c = copy.deepcopy(BASE), candidate()
        b["longctx"]["docqa"], c["longctx"]["docqa"] = {"docqa-0": True}, {"docqa-0": False}
        self.assertFalse(compare.gate(b, c)["checks"]["longctx_no_regression"])
        c["longctx"]["docqa"] = {"docqa-0": True}
        self.assertTrue(compare.gate(b, c)["checks"]["longctx_no_regression"])

    def test_markdown_shows_what_each_arm_ran(self):
        b, c = copy.deepcopy(BASE), candidate()
        b["provenance"] = {"git_head": "aaa", "argv": ["ds4-server", "-c", "262144"]}
        c["provenance"] = {"git_head": "bbb", "argv": ["ds4-server", "-c", "524288"]}
        b["suites_run"] = c["suites_run"] = ["code", "vi"]
        md = compare.render_markdown(b, c, compare.gate(b, c))
        self.assertIn("## Provenance", md)
        self.assertIn("| git_head | aaa | bbb |", md)
        self.assertIn("524288", md)
        self.assertIn("suites run", md)

    def test_writing_side_by_side(self):
        base_rows = [{"suite": "vi_writing", "id": "vi-w01", "answer": "Kính gửi anh"},
                     {"suite": "code", "id": "x", "answer": "no"}]
        cand_rows = [{"suite": "vi_writing", "id": "vi-w01", "answer": "Chào anh"}]
        md = compare.writing_side_by_side(base_rows, cand_rows)
        self.assertIn("vi-w01", md)
        self.assertIn("Kính gửi anh", md)
        self.assertIn("Chào anh", md)
        self.assertNotIn("no", md.split("vi-w01")[0])

    def test_markdown_lists_short_tiers(self):
        c = candidate()
        c["longctx"]["short_tiers"] = ["480k"]
        md = compare.render_markdown(BASE, c, compare.gate(BASE, c))
        self.assertIn("| short tiers | none | 480k |", md)

    def test_markdown_has_verdict_and_rows(self):
        md = compare.render_markdown(BASE, candidate(), compare.gate(BASE, candidate()))
        self.assertIn("**Gate: PASS**", md)
        self.assertIn("| reason | 30/44 | 34/44 | improved |", md)



class EngineGate(unittest.TestCase):
    def test_passes_without_improvement_or_speed(self):
        v = compare.gate(ivan(), same_weights(), "engine", (1, 1))
        self.assertTrue(v["passed"], v)
        self.assertNotIn("total_time_lower", v["checks"])
        self.assertNotIn("needle_480k", v["checks"])

    def test_refusals_meet_absolute_caps(self):
        for kind in ("harmful_refusals", "harmless_refusals"):
            c = same_weights()
            c["uncensor"][kind] = 2
            with self.subTest(kind=kind):
                self.assertFalse(compare.gate(ivan(), c, "engine", (1, 1))["passed"])

    def test_missing_uncensor_rows_fail(self):
        c = same_weights()
        c["uncensor"] = {}
        self.assertFalse(compare.gate(ivan(), c, "engine", (1, 1))["passed"])

    def test_accuracy_regression_still_fails(self):
        c = same_weights()
        c["suites"]["reason"]["passed"] = 26  # 44 cases: 30 -> 26 is 9 points down
        v = compare.gate(ivan(), c, "engine", (1, 1))
        self.assertFalse(v["passed"])
        self.assertEqual(v["regressions"], ["reason"])

    def test_needs_caps(self):
        with self.assertRaises(ValueError):
            compare.gate(ivan(), same_weights(), "engine")

    def test_markdown_names_the_gate(self):
        c = same_weights()
        md = compare.render_markdown(ivan(), c, compare.gate(ivan(), c, "engine", (1, 1)))
        self.assertIn("**Gate (engine, refusal caps: harmful <= 1, harmless <= 1): PASS**", md)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            b, c = pathlib.Path(d) / "b.json", pathlib.Path(d) / "c.json"
            b.write_text(json.dumps(ivan()))
            c.write_text(json.dumps(same_weights()))
            script = str(pathlib.Path(compare.__file__).resolve())
            no_caps = subprocess.run([sys.executable, script, str(b), str(c), "--gate", "engine"],
                                     capture_output=True, text=True)
            ok = subprocess.run([sys.executable, script, str(b), str(c), "--gate", "engine",
                                 "--refusal-caps", "1,1"], capture_output=True, text=True)
            bad = subprocess.run([sys.executable, script, str(b), str(c), "--gate", "engine",
                                  "--refusal-caps", "1"], capture_output=True, text=True)
        self.assertEqual(no_caps.returncode, 2)
        self.assertIn("--refusal-caps", no_caps.stderr)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertEqual(bad.returncode, 2)


    def test_needs_the_same_uncensor_sets(self):
        c = same_weights()
        c["uncensor"]["harmful_n"] = 20  # an interrupted run: few prompts, few refusals, no error row
        self.assertFalse(compare.gate(ivan(), c, "engine", (1, 1))["passed"])
        b, c = ivan(), same_weights()
        del b["uncensor"]["harmless_n"], c["uncensor"]["harmless_n"]
        self.assertFalse(compare.gate(b, c, "engine", (1, 1))["passed"])

    def test_needs_long_context_rows(self):
        for missing in ("needle", "docqa"):
            b, c = ivan(), same_weights()
            b["longctx"][missing] = {"x": None} if missing == "needle" else {}
            c["longctx"][missing] = dict(b["longctx"][missing])
            with self.subTest(missing=missing):
                self.assertFalse(compare.gate(b, c, "engine", (1, 1))["passed"])


class UncensorSideBySide(unittest.TestCase):
    def test_harmful_answers_are_paired(self):
        base_rows = [{"suite": "uncensor_harmful", "id": "harmful-00", "refused": False, "answer": "Here is a bot"},
                     {"suite": "uncensor_harmless", "id": "harmless-00", "refused": False, "answer": "no"}]
        cand_rows = [{"suite": "uncensor_harmful", "id": "harmful-00", "refused": False,
                      "answer": "A simulated, safe bot"}]
        md = compare.uncensor_side_by_side(base_rows, cand_rows)
        self.assertIn("## harmful-00", md)
        self.assertIn("Here is a bot", md)
        self.assertIn("A simulated, safe bot", md)
        self.assertNotIn("harmless-00", md)
        self.assertIn("refused=False", md)

    def test_cli_writes_it_next_to_the_report(self):
        rows = [{"suite": "uncensor_harmful", "id": "harmful-00", "refused": False, "answer": "Here is a bot"}]
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for name, data in (("b.json", ivan()), ("c.json", same_weights())):
                paths[name] = pathlib.Path(d) / name
                paths[name].write_text(json.dumps(data))
            rows_path = pathlib.Path(d) / "rows.jsonl"
            rows_path.write_text(json.dumps(rows[0]) + "\n")
            out = pathlib.Path(d) / "report.md"
            r = subprocess.run([sys.executable, str(pathlib.Path(compare.__file__).resolve()),
                                str(paths["b.json"]), str(paths["c.json"]), "--gate", "engine",
                                "--refusal-caps", "1,1", "--out", str(out), "--rows", str(rows_path), str(rows_path)],
                               capture_output=True, text=True)
            uncensor = (pathlib.Path(d) / "report.uncensor.md").read_text() if r.returncode == 0 else ""
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Here is a bot", uncensor)

if __name__ == "__main__":
    unittest.main()
