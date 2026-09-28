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
    lines = ["ds4-eval: %d/%d passed, runtime 0:10" % (sum(r[1] == "PASSED" for r in rows), len(rows)),
             "%-3s %-10s %8s %8s %8s %-20s %-20s %s" % ("#", "state", "prompt", "gen", "total", "given",
                                                        "correct", "test")]
    for i, (source, state, given, correct) in enumerate(rows, 1):
        lines.append("%3d %-10s %8d %8d %8d %-20.20s %-20.20s %s/%s" % (
            i, state, 100, 200, 300, given, correct, source, "case-%d" % i))
    return "\n".join(lines) + "\n"


class Argv(unittest.TestCase):
    def test_eval_argv_whitelists_and_caps_ctx(self):
        argv = ds4eval.eval_argv(SERVER_ARGV, pathlib.Path("/repo"), "core", "GPQA Diamond", 8, "/t/x.trace")
        self.assertEqual(argv, [
            "/repo/ds4-eval", "--plain", "--metal", "-m", "/m/prod.gguf", "--ple", "/m/ple.gguf",
            "--prefill-chunk", "2048", "--ssd-streaming", "--ssd-streaming-cache-experts", "6GB",
            "-c", "65536", "--suite", "core", "--source", "GPQA Diamond", "--questions", "8",
            "--tokens", "32768", "--trace", "/t/x.trace"])


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
        text = "loading model...\n" + report([("AIME2025", "FAILED", "12", "70")]) + "done\n"
        self.assertEqual(len(ds4eval.parse_report(text)), 1)


class Run(unittest.TestCase):
    def _fake_run(self, missing_rows=False):
        def fake(argv, **kwargs):
            n = int(argv[argv.index("--questions") + 1])
            source = argv[argv.index("--source") + 1]
            count = n - 1 if missing_rows else n
            rows = [(source, "PASSED" if i % 2 == 0 else "FAILED", "A", "A") for i in range(count)]
            return mock.Mock(stdout=report(rows), stderr="", returncode=0)
        return fake

    def test_run_reason_collects_every_run(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ds4eval.subprocess, "run", self._fake_run()):
            rows = ds4eval.run_reason({}, SERVER_ARGV, pathlib.Path("/repo"), d)
        self.assertEqual(len(rows), sum(n for _, _, n in ds4eval.REASON_RUNS))
        self.assertTrue(all(r["suite"] == "reason" for r in rows))
        self.assertEqual(rows[0]["passed"], True)
        self.assertEqual(rows[1]["passed"], False)

    def test_short_report_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(ds4eval.subprocess, "run", self._fake_run(missing_rows=True)):
            with self.assertRaises(RuntimeError):
                ds4eval.run_reason({}, SERVER_ARGV, pathlib.Path("/repo"), d)


if __name__ == "__main__":
    unittest.main()
