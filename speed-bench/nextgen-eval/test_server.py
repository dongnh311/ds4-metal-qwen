import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import server  # noqa: E402

PROD_CMD = ["/usr/bin/env", "DS4_QWEN4_STREAM_FULL_LAYERS=32", "DS4_QWEN4_KV_GROW=1",
            "/opt/ds4/ds4-server", "--metal", "-m", "/m/prod.gguf", "--ple", "/m/ple.gguf",
            "-c", "262144", "--prefill-chunk", "2048", "--mtp", "--ssd-streaming",
            "--ssd-streaming-cache-experts", "6GB", "--kv-disk-dir", "/kv/prod",
            "--kv-disk-space-mb", "32768", "--host", "127.0.0.1", "--port", "18086"]
REGISTRY = {"models": {
    "a": {"runtimes": {"ds4": {"enabled": True, "process_command": PROD_CMD}}},
    "b": {"runtimes": {"ds4": {"enabled": False, "process_command": []}}},
    "c": {"runtimes": {"omlx": {"enabled": True}}},
}}
LOG = """0925 10:18:05 ds4-server: chat ctx=0..81:81 prompt start
0925 10:18:06 ds4-server: chat ctx=0..81:81 prefill chunk 81/81 (100.0%) chunk=0.00 t/s avg=83.43 t/s 0.971s
0925 10:18:06 ds4-server: chat ctx=0..81:81 prompt done 0.971s
0925 10:18:07 ds4-server: chat ctx=81..131:50 gen=50 THINKING decoding chunk=41.61 t/s avg=41.61 t/s 1.202s
0925 10:18:10 ds4-server: chat ctx=0..81:81 thinking closed after 172 tokens
0925 10:18:13 ds4-server: chat ctx=331..370:39 gen=289 decoding chunk=41.66 t/s avg=43.72 t/s 6.610s
0925 10:18:13 ds4-server: chat ctx=0..81:81 gen=289 finish=stop 7.581s
"""
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12345.
Pages wired down:                        %d.
Swapins:                                      0.
Swapouts:                                    %d.
"""


class Resolve(unittest.TestCase):
    def test_split_env(self):
        env, argv = server.split_env(PROD_CMD)
        self.assertEqual(env, {"DS4_QWEN4_STREAM_FULL_LAYERS": "32", "DS4_QWEN4_KV_GROW": "1"})
        self.assertEqual(argv[0], "/opt/ds4/ds4-server")

    def test_resolve_prod_arm(self):
        cfg = {"name": "prod", "base": "registry", "model": None, "args_add": [], "args_remove": [], "env": {}}
        env, argv = server.resolve(cfg, REGISTRY, pathlib.Path("/repo"), 18299, "/tmp/kv")
        self.assertEqual(argv[0], "/repo/ds4-server")
        self.assertEqual(server.argv_value(argv, "--port"), "18299")
        self.assertEqual(server.argv_value(argv, "--kv-disk-dir"), "/tmp/kv")
        self.assertEqual(server.argv_value(argv, "-m"), "/m/prod.gguf")
        self.assertEqual(env["DS4_QWEN4_KV_GROW"], "1")

    def test_resolve_overrides(self):
        cfg = {"name": "cand", "base": "registry", "model": "/m/new.gguf",
               "args_remove": ["--mtp", "--ssd-streaming-cache-experts"],
               "args_add": ["--refusal-projection", "/d/v.gguf"], "env": {"DS4_QWEN4_KV_GROW": "0"}}
        env, argv = server.resolve(cfg, REGISTRY, pathlib.Path("/repo"), 18299, "/tmp/kv")
        self.assertEqual(server.argv_value(argv, "-m"), "/m/new.gguf")
        self.assertNotIn("--mtp", argv)
        self.assertNotIn("--ssd-streaming-cache-experts", argv)
        self.assertNotIn("6GB", argv)
        self.assertIn("--ssd-streaming", argv)
        self.assertEqual(argv[-2:], ["--refusal-projection", "/d/v.gguf"])
        self.assertEqual(env["DS4_QWEN4_KV_GROW"], "0")

    def test_registry_needs_exactly_one_enabled_ds4(self):
        two = {"models": {"a": REGISTRY["models"]["a"], "x": REGISTRY["models"]["a"]}}
        with self.assertRaises(SystemExit):
            server.registry_command(two)

    def test_registry_model_picks_among_several_enabled(self):
        # Ornith's ds4 runtime is enabled next to PROD's Qwen (2026-09-28).
        two = {"models": {"a": REGISTRY["models"]["a"],
                          "ornith": {"runtimes": {"ds4": {"enabled": True, "process_command": ["x"]}}}}}
        self.assertEqual(server.registry_command(two, "a"), PROD_CMD)
        with self.assertRaises(SystemExit) as cm:
            server.registry_command(two)
        self.assertIn("ornith", str(cm.exception))
        cfg = {"name": "prod", "base": "registry", "registry_model": "a"}
        env, argv = server.resolve(cfg, two, pathlib.Path("/repo"), 18299, "/tmp/kv")
        self.assertEqual(server.argv_value(argv, "-m"), "/m/prod.gguf")

    def test_args_add_replaces_a_flag_the_registry_has(self):
        cfg = {"name": "cand", "base": "registry", "args_add": ["-c", "524288", "--warm-weights"]}
        env, argv = server.resolve(cfg, REGISTRY, pathlib.Path("/repo"), 18299, "/tmp/kv")
        self.assertEqual(argv.count("-c"), 1)
        self.assertEqual(server.argv_value(argv, "-c"), "524288")
        self.assertEqual(argv[-1], "--warm-weights")

    def test_ctx_alias_is_normalized(self):
        cfg = {"name": "cand", "base": "registry", "args_add": ["--ctx", "524288"]}
        env, argv = server.resolve(cfg, REGISTRY, pathlib.Path("/repo"), 18299, "/tmp/kv")
        self.assertNotIn("--ctx", argv)
        self.assertEqual(server.argv_value(argv, "-c"), "524288")

    def test_unknown_base_is_refused(self):
        with self.assertRaises(SystemExit):
            server.resolve({"name": "x", "base": "file"}, REGISTRY, pathlib.Path("/repo"), 1, "/k")


class Logs(unittest.TestCase):
    def test_parse_request_log(self):
        self.assertEqual(server.parse_request_log(LOG), {
            "prefill_s": 0.971, "prefill_tps": 83.43, "think_tokens": 172, "gen_tokens": 289,
            "finish": "stop", "total_s": 7.581, "decode_tps": 43.72})

    def test_parse_request_log_without_thinking(self):
        seg = "\n".join(l for l in LOG.splitlines() if "thinking closed" not in l)
        self.assertIsNone(server.parse_request_log(seg)["think_tokens"])

    def test_parse_empty_segment(self):
        self.assertTrue(all(v is None for v in server.parse_request_log("").values()))

    def test_log_cursor_returns_only_new_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "s.log"
            cur = server.LogCursor(p)
            self.assertEqual(cur.take(), "")
            p.write_text("one\n")
            self.assertEqual(cur.take(), "one\n")
            with open(p, "a") as f:
                f.write("two\n")
            self.assertEqual(cur.take(), "two\n")


class Memory(unittest.TestCase):
    def test_parse_vm_stat(self):
        s = server.parse_vm_stat(VM_STAT % (200000, 42))
        self.assertEqual(s, {"page_size": 16384, "wired_pages": 200000, "swapouts": 42})

    def test_sampler_peak_and_swapouts(self):
        texts = iter([VM_STAT % (100000, 10), VM_STAT % (300000, 10), VM_STAT % (200000, 12)])
        sampler = server.MemSampler(interval=3600, read=lambda: next(texts))
        sampler.start()
        sampler._sample()
        self.assertEqual(sampler.stop(), {"peak_wired_gib": 4.58, "swapouts": 2})


class Stop(unittest.TestCase):
    def test_stop_never_sigkills(self):
        prog = ("import signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "print('ready', flush=True)\n"
                "time.sleep(60)\n")
        proc = subprocess.Popen([sys.executable, "-c", prog], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), "ready")
            srv = server.Ds4Server({}, ["unused"], "/dev/null", 1)
            srv.proc = proc
            with self.assertRaises(RuntimeError):
                srv.stop(term_timeout=1, drain=0)
            self.assertIsNone(proc.poll())
        finally:
            proc.kill()
            proc.wait()
            proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
