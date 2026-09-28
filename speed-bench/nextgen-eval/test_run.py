import hashlib
import json
import pathlib
import sys
import tempfile
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
        self.assertEqual(s["longctx"], {"needle": {"120k": True, "480k": None}, "docqa": {},
                                        "peak_wired_gib": 49.1, "swapouts": 0})
        self.assertEqual(s["speed"]["total_seconds"], {"code": 30.0, "vi": 12.0, "uncensor": 5.0})
        self.assertEqual(s["speed"]["think_tokens_median"], 200)
        self.assertEqual(s["speed"]["decode_tps_median"], 41.0)

    def test_empty_rows(self):
        s = run.summarize([])
        self.assertEqual(s["suites"], {})
        self.assertIsNone(s["vi_cjk_leaks"])

    def test_prod_config(self):
        cfg = json.loads((HERE / "configs" / "prod.json").read_text())
        self.assertEqual(cfg, {"name": "prod", "base": "registry",
                               "registry_model": "ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2",
                               "model": None, "args_add": [], "args_remove": [], "env": {}})

    def test_case_ids_and_doc_questions(self):
        rows = ROWS + [{"suite": "longctx", "id": "docqa-0", "passed": False}]
        s = run.summarize(rows)
        self.assertEqual(s["case_ids"]["code"], hashlib.sha256(b"a\nb").hexdigest()[:16])
        self.assertNotIn("vi_writing", s["case_ids"])
        self.assertEqual(s["longctx"]["docqa"], {"docqa-0": False})
        self.assertEqual(s["longctx"]["needle"], {"120k": True, "480k": None})

    def test_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root, data = pathlib.Path(d) / "repo", pathlib.Path(d) / "data"
            root.mkdir()
            data.mkdir()
            (root / "ds4-server").write_bytes(b"server")
            (data / "manifest.json").write_text('{"mbpp.jsonl": {"rows": 50, "sha256": "abc"}}')
            calls = []

            def git(repo, *args):
                calls.append((str(repo), args))
                return {"rev-parse": "c0ffee", "status": " M ds4.c"}[args[0]]
            p = run.provenance({"K": "1"}, ["ds4-server", "-c", "8"], root, data, pathlib.Path(d) / "gw", git=git)
        self.assertEqual(p["git_head"], "c0ffee")
        self.assertTrue(p["git_dirty"])
        self.assertEqual(p["gateway_head"], "c0ffee")
        self.assertEqual(p["ds4_server_sha256"], hashlib.sha256(b"server").hexdigest())
        self.assertIsNone(p["ds4_eval_sha256"])
        self.assertEqual(p["data"], {"mbpp.jsonl": "abc"})
        self.assertEqual((p["env"], p["argv"]), ({"K": "1"}, ["ds4-server", "-c", "8"]))

    def test_errors_are_listed(self):
        s = run.summarize(ROWS + [{"suite": "tools", "id": "suite-error", "passed": None, "error": "boom"}])
        self.assertEqual(s["errors"], [{"suite": "tools", "error": "boom"}])

    def test_all_suites(self):
        self.assertEqual(run.ALL_SUITES, ["code", "ifeval", "vi", "uncensor", "tools", "longctx", "reason"])


class FakeProc:
    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


class FakeServer:
    instances = []

    def __init__(self, env, argv, log_path, port):
        self.base_url, self.proc, self.stopped = "http://fake", FakeProc(), False
        FakeServer.instances.append(self)

    def start(self):
        pass

    def stop(self):
        self.stopped = True


class RunArm(unittest.TestCase):
    ARGV = ["/repo/ds4-server", "-c", "262144"]

    def _arm(self, suites, wanted, reason=None):
        FakeServer.instances = []
        rows = []
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d)
            run.run_arm({}, self.ARGV, wanted, out, rows, server_cls=FakeServer, suites=suites,
                        reason=reason or (lambda env, argv, root, out_dir: iter([{"suite": "reason", "id": "r",
                                                                                   "passed": True}])))
            on_disk = [json.loads(l) for l in (out / "rows.jsonl").read_text().splitlines()]
        return rows, on_disk

    def test_a_failing_suite_keeps_its_rows_and_the_run_continues(self):
        def flaky(ctx):
            yield {"suite": "a", "id": "a1", "passed": True}
            raise OSError("connection reset")

        def fine(ctx):
            yield {"suite": "b", "id": "b1", "passed": True}
        rows, on_disk = self._arm({"a": flaky, "b": fine}, ["a", "b", "reason"])
        self.assertEqual([(r["suite"], r["id"]) for r in on_disk],
                         [("a", "a1"), ("a", "suite-error"), ("b", "b1"), ("reason", "r")])
        self.assertIn("connection reset", on_disk[1]["error"])
        self.assertEqual(rows, on_disk)
        self.assertTrue(FakeServer.instances[0].stopped)

    def test_unknown_flag_fails_before_the_server_starts(self):
        FakeServer.instances = []
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            run.run_arm({}, self.ARGV + ["--refusal-projection", "/d/v.gguf"], ["a", "reason"], pathlib.Path(d),
                        [], server_cls=FakeServer, suites={"a": lambda ctx: iter([])})
        self.assertEqual(FakeServer.instances, [])

    def test_a_dead_server_ends_the_server_suites(self):
        def dies(ctx):
            FakeServer.instances[0].proc.code = -6
            yield {"suite": "a", "id": "a1", "passed": True}

        def never(ctx):
            raise AssertionError("must not run after the server died")
        rows, _ = self._arm({"a": dies, "b": never}, ["a", "b"])
        self.assertEqual([(r["suite"], r["id"]) for r in rows], [("a", "a1"), ("b", "suite-error")])
        self.assertIn("exited", rows[1]["error"])


if __name__ == "__main__":
    unittest.main()
