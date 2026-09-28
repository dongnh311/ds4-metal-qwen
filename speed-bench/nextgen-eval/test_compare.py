import copy
import pathlib
import sys
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

    def test_markdown_has_verdict_and_rows(self):
        md = compare.render_markdown(BASE, candidate(), compare.gate(BASE, candidate()))
        self.assertIn("**Gate: PASS**", md)
        self.assertIn("| reason | 30/44 | 34/44 | improved |", md)


if __name__ == "__main__":
    unittest.main()
