import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ds4eval  # noqa: E402

SERVER_ARGV = ["/repo/ds4-server", "--metal", "-m", "/m/prod.gguf", "--ple", "/m/ple.gguf",
               "-c", "262144", "--prefill-chunk", "2048", "--mtp", "--ssd-streaming",
               "--ssd-streaming-cache-experts", "6GB", "--kv-disk-dir", "/kv", "--kv-disk-space-mb",
               "32768", "--host", "127.0.0.1", "--port", "18299"]


def report(rows):
    """A ds4-eval report as bytes, padded the way C's %-20.20s pads: by bytes, not characters."""
    lines = [b"ds4-eval: %d/%d passed, runtime 0:10" % (sum(r[1] == "PASSED" for r in rows), len(rows)),
             b"#   state        prompt      gen    total given                correct              test"]
    for i, (source, state, given, correct) in enumerate(rows, 1):
        lines.append(b"%3d %-10s %8d %8d %8d %s %s %s/%s" % (
            i, state.encode(), 100, 200, 300, given.encode().ljust(20)[:20], correct.encode().ljust(20)[:20],
            source.encode(), b"case-%d" % i))
    return b"\n".join(lines) + b"\n"


class FakePopen:
    """Stands in for subprocess.Popen: `script(argv)` returns (returncode, stdout bytes) or raises."""
    script = None
    last = None

    def __init__(self, argv, **kwargs):
        self.argv, self.returncode, self.terminated, self.killed = argv, None, False, False
        FakePopen.last = self

    def communicate(self, timeout=None):
        self.returncode, out = FakePopen.script(self.argv)
        return out, b""

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode


class Argv(unittest.TestCase):
    def test_eval_argv_whitelists_and_caps_ctx(self):
        argv = ds4eval.eval_argv(SERVER_ARGV, pathlib.Path("/repo"), "core", "GPQA Diamond", 8, "/t/x.trace")
        self.assertEqual(argv, [
            "/repo/ds4-eval", "--plain", "--metal", "-m", "/m/prod.gguf", "--ple", "/m/ple.gguf",
            "--prefill-chunk", "2048", "--ssd-streaming", "--ssd-streaming-cache-experts", "6GB",
            "-c", "65536", "--suite", "core", "--source", "GPQA Diamond", "--questions", "8",
            "--tokens", "32768", "--trace", "/t/x.trace"])

    def test_server_only_flags_are_dropped(self):
        argv = SERVER_ARGV + ["--kv-cache-cold-max-tokens", "262144", "--think-budget", "4096"]
        self.assertEqual(ds4eval.eval_argv(argv, pathlib.Path("/repo"), "core", "AIME2025", 8, "/t"),
                         ds4eval.eval_argv(SERVER_ARGV, pathlib.Path("/repo"), "core", "AIME2025", 8, "/t"))

    def test_unknown_flag_is_refused(self):
        # A candidate flag (say, runtime refusal projection) must never be dropped silently: the
        # reasoning suite would score a model other than the candidate.
        with self.assertRaises(ValueError) as cm:
            ds4eval.eval_argv(SERVER_ARGV + ["--refusal-projection", "/d/v.gguf"], pathlib.Path("/repo"),
                              "core", "AIME2025", 8, "/t")
        self.assertIn("--refusal-projection", str(cm.exception))


class Report(unittest.TestCase):
    def test_parse_fixed_columns_with_spaces(self):
        rows = ds4eval.parse_report(report([("GPQA Diamond", "PASSED", "B (and C)", "B"),
                                            ("GPQA Diamond", "INCOMPLETE", "-", "D")]))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["given"], "B (and C)")
        self.assertEqual(rows[0]["source"], "GPQA Diamond")
        self.assertEqual(rows[0]["case_id"], "case-1")
        self.assertEqual(rows[1]["state"], "INCOMPLETE")

    def test_ignores_progress_lines(self):
        text = b"loading model...\n" + report([("AIME2025", "FAILED", "12", "70")]) + b"done\n"
        self.assertEqual(len(ds4eval.parse_report(text)), 1)


class Run(unittest.TestCase):
    def _run(self, script):
        FakePopen.script = staticmethod(script)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ds4eval.subprocess, "Popen", FakePopen):
            return list(ds4eval.run_reason({}, SERVER_ARGV, pathlib.Path("/repo"), d))

    @staticmethod
    def _rows(argv, count_delta=0, states=None):
        n = int(argv[argv.index("--questions") + 1])
        source = argv[argv.index("--source") + 1]
        states = states or ["PASSED" if i % 2 == 0 else "FAILED" for i in range(n)]
        return [(source, states[i], "A", "A") for i in range(n + count_delta)]

    def test_run_reason_collects_every_run(self):
        rows = self._run(lambda argv: (1, report(self._rows(argv))))
        self.assertEqual(len(rows), sum(n for _, _, n in ds4eval.REASON_RUNS))
        self.assertTrue(all(r["suite"] == "reason" for r in rows))
        self.assertEqual(rows[0]["passed"], True)
        self.assertEqual(rows[1]["passed"], False)

    def test_finished_runs_stream_before_a_later_failure(self):
        def fourth_fails(argv):
            if "MMLU-Pro" in argv:
                return 1, b"engine error\n"
            return 1, report(self._rows(argv))
        got = []
        FakePopen.script = staticmethod(fourth_fails)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ds4eval.subprocess, "Popen", FakePopen):
            with self.assertRaises(RuntimeError):
                for row in ds4eval.run_reason({}, SERVER_ARGV, pathlib.Path("/repo"), d):
                    got.append(row)
        self.assertEqual(len(got), 8 + 8 + 8)

    def test_short_report_fails_loudly(self):
        with self.assertRaises(RuntimeError):
            self._run(lambda argv: (0, report(self._rows(argv, count_delta=-1))))

    def test_engine_error_report_fails_loudly(self):
        # ds4-eval stops on an engine error but still prints one row per case, then exits 1.
        def engine_error(argv):
            n = int(argv[argv.index("--questions") + 1])
            states = ["PASSED", "FAILED", "RUNNING"] + ["PENDING"] * (n - 3)
            return 1, report(self._rows(argv, states=states))
        with self.assertRaises(RuntimeError):
            self._run(engine_error)

    def test_exit_one_with_failed_cases_is_normal(self):
        # ds4_eval.c returns `rc || failed || incomplete ? 1 : 0`: one wrong answer exits 1 (seen live on
        # SuperGPQA 7/8, 2026-09-28).
        rows = self._run(lambda argv: (1, report(self._rows(argv))))
        self.assertEqual(len(rows), sum(n for _, _, n in ds4eval.REASON_RUNS))

    def test_exit_zero_with_failed_cases_fails_loudly(self):
        with self.assertRaises(RuntimeError):
            self._run(lambda argv: (0, report(self._rows(argv))))

    def test_nonzero_exit_with_every_case_passed_fails_loudly(self):
        def all_passed(argv):
            n = int(argv[argv.index("--questions") + 1])
            return 1, report(self._rows(argv, states=["PASSED"] * n))
        with self.assertRaises(RuntimeError):
            self._run(all_passed)

    def test_skipped_case_fails_loudly(self):
        def skipped(argv):
            n = int(argv[argv.index("--questions") + 1])
            return 0, report(self._rows(argv, states=["SKIPPED"] + ["PASSED"] * (n - 1)))
        with self.assertRaises(RuntimeError):
            self._run(skipped)

    def test_interrupt_terminates_and_never_kills(self):
        def interrupted(argv):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self._run(interrupted)
        self.assertTrue(FakePopen.last.terminated)
        self.assertFalse(FakePopen.last.killed)

    def test_partial_utf8_output_and_non_ascii_given(self):
        # --plain streams generated tokens to stdout; one can end inside a multi-byte character.
        def odd_bytes(argv):
            rows = [(r[0], r[1], "\u221a2 (sqrt)", "A") for r in self._rows(argv)]
            return 1, b"token stream \xe2\x88\n" + report(rows)
        rows = self._run(odd_bytes)
        self.assertEqual(rows[0]["given"], "\u221a2 (sqrt)")


if __name__ == "__main__":
    unittest.main()
