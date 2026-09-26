"""Unit tests for the Ornith M4 A/B harness plumbing (python3 -m unittest).

No model, no GPU: the process guards, prompt building, cached-tokens assertion,
one-process-at-a-time discipline, the ds4-vs-ds4 lever mode and the gate-3
verdict are checked with fakes.
"""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))                       # speed-bench/ornith
sys.path.insert(0, os.path.join(HERE, "..", "..", "qwen-regression"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import m4_ab as m  # noqa: E402


class PromptTest(unittest.TestCase):
    def test_nonce_is_prefix(self):
        p = m.build_prompt("nonce-abc", 2048, "x" * 200000, 3.46)
        self.assertTrue(p.startswith("nonce-abc"))

    def test_target_tokens_scales_chars(self):
        short = m.build_prompt("n", 2048, "x" * 200000, 3.46)
        long = m.build_prompt("n", 32768, "x" * 2000000, 3.46)
        self.assertLess(len(short), len(long))
        # ~3.46 chars/token: a 2048-token prompt is roughly 7K chars, well under 200K
        self.assertLess(len(short), 20000)


class GuardTest(unittest.TestCase):
    def test_one_process_guard_refuses_when_other_up(self):
        # ds4_running returns a non-empty string => refuse
        with self.assertRaises(SystemExit):
            m.guard_free(ds4_running=lambda: "12345 ds4-server", omlx_running=lambda: "",
                         estate=lambda: "")

    def test_one_process_guard_refuses_when_omlx_up(self):
        with self.assertRaises(SystemExit):
            m.guard_free(ds4_running=lambda: "", omlx_running=lambda: "40263 omlx-server",
                         estate=lambda: "")

    def test_one_process_guard_refuses_on_estate(self):
        with self.assertRaises(SystemExit):
            m.guard_free(ds4_running=lambda: "", omlx_running=lambda: "",
                         estate=lambda: "999 E hung")

    def test_one_process_guard_passes_when_free(self):
        m.guard_free(ds4_running=lambda: "", omlx_running=lambda: "", estate=lambda: "")


class CachedTokensTest(unittest.TestCase):
    def test_cached_tokens_zero_asserted(self):
        usage = {"prompt_tokens_details": {"cached_tokens": 7}}
        with self.assertRaises(m.CacheHit):
            m.assert_cache_cold(usage)

    def test_cached_tokens_zero_ok(self):
        m.assert_cache_cold({"prompt_tokens_details": {"cached_tokens": 0}})
        m.assert_cache_cold({"prompt_tokens": 100})  # no details => treated as 0


class SwapTest(unittest.TestCase):
    def test_swap_delta_recorded(self):
        rec = m.measure_arm(
            _FakeArm(), contexts=[2048], filler="x" * 100000, max_tokens=8, warmup=0,
            swap_used=iter([100.0, 140.0]).__next__,
            wait_idle=lambda: None)
        self.assertEqual(rec["swap_before_mib"], 100.0)
        self.assertEqual(rec["swap_after_mib"], 140.0)
        self.assertEqual(rec["swap_delta_mib"], 40.0)


class FinishReasonTest(unittest.TestCase):
    def test_finish_reason_recorded_when_present(self):
        class _Arm:
            name = "ds4"

            def stream(self, prompt, max_tokens):
                return ({"prompt_tokens": 2048, "completion_tokens": 8, "ttft_s": 1.0,
                         "decode_s": 0.1, "finish_reason": "stop"},
                        {"prompt_tokens_details": {"cached_tokens": 0}})

        rec = m.measure_arm(_Arm(), contexts=[2048], filler="x" * 100000, max_tokens=8, warmup=0,
                            swap_used=iter([0.0, 0.0]).__next__, wait_idle=lambda: None)
        self.assertEqual(rec["rows"]["2048"]["finish_reason"], "stop")

    def test_finish_reason_defaults_to_none_when_absent(self):
        rec = m.measure_arm(_FakeArm(), contexts=[2048], filler="x" * 100000, max_tokens=8, warmup=0,
                            swap_used=iter([0.0, 0.0]).__next__, wait_idle=lambda: None)
        self.assertIsNone(rec["rows"]["2048"]["finish_reason"])


class TerminateTest(unittest.TestCase):
    """Fix 16: arms never SIGKILL; a survivor after SIGTERM is left running."""

    def test_terminate_returns_when_process_exits_cleanly(self):
        proc = _CleanProc()
        m._terminate(proc, "ds4:test")
        self.assertTrue(proc.terminated)

    def test_terminate_raises_naming_pid_when_process_survives(self):
        proc = _StuckProc()
        with self.assertRaises(SystemExit) as cm:
            m._terminate(proc, "ds4:test")
        self.assertIn("4242", str(cm.exception))

    def test_terminate_never_sends_sigkill(self):
        proc = _StuckProc()
        with self.assertRaises(SystemExit):
            m._terminate(proc, "ds4:test")
        # _StuckProc.kill() raises AssertionError if ever called; reaching
        # the SystemExit above without that error proves kill() was unused.


class LeverModeTest(unittest.TestCase):
    def test_lever_order_is_abba(self):
        self.assertEqual(m.LEVER_ORDER, ("base", "lever", "lever", "base"))

    def test_baseline_order_is_abba(self):
        self.assertEqual(m.BASELINE_ORDER, ("omlx", "ds4", "ds4", "omlx"))

    def test_interleave_lever_visits_arms_in_abba_order(self):
        seen = []

        def make_base():
            seen.append("base")
            return _FakeArm(name="base")

        def make_lever():
            seen.append("lever")
            return _FakeArm(name="lever")

        m.interleave({"base": make_base, "lever": make_lever}, contexts=[2048], cold_tokens=0,
                    filler="x" * 100000, max_tokens=8, warmup=0, order=m.LEVER_ORDER,
                    guard=lambda: None, wait_free=lambda: None,
                    swap_used=iter([0.0] * 8).__next__, wait_idle=lambda: None)
        self.assertEqual(seen, ["base", "lever", "lever", "base"])

    def test_lever_env_reaches_only_lever_arm_process_environment(self):
        calls = []

        class _FakePopen:
            def __init__(self, cmd, stdout=None, stderr=None, env=None):
                calls.append(env)
                self.pid = 1

            def poll(self):
                return None

        orig_popen = m.subprocess.Popen
        m.subprocess.Popen = _FakePopen
        try:
            base_arm = m.Ds4Arm("model.gguf", _tmp_out(), extra_env={}, label="base")
            lever_arm = m.Ds4Arm("model.gguf", _tmp_out(), extra_env={"DS4_LEVER": "1"},
                                 label="lever")
            base_arm.start()
            lever_arm.start()
        finally:
            m.subprocess.Popen = orig_popen
        self.assertNotIn("DS4_LEVER", calls[0])
        self.assertIn("DS4_LEVER", calls[1])
        self.assertEqual(calls[1]["DS4_LEVER"], "1")


class ParseEnvTest(unittest.TestCase):
    def test_parse_env_splits_pairs(self):
        self.assertEqual(m._parse_env("A=1,B=2"), {"A": "1", "B": "2"})

    def test_parse_env_empty_string(self):
        self.assertEqual(m._parse_env(""), {})


class VerdictTest(unittest.TestCase):
    def test_pass_when_ds4_faster_everywhere(self):
        summary = {"2048": {"ds4_decode": 90, "omlx_decode": 65, "ds4_prefill": 2100, "omlx_prefill": 2000},
                   "cold31k": {"ds4_ttft_s": 14.0, "omlx_ttft_s": 15.2}}
        self.assertEqual(m.verdict(summary), [])

    def test_fail_when_ds4_decode_slower(self):
        summary = {"2048": {"ds4_decode": 60, "omlx_decode": 65, "ds4_prefill": 2100, "omlx_prefill": 2000},
                   "cold31k": {"ds4_ttft_s": 14.0, "omlx_ttft_s": 15.2}}
        self.assertTrue(any("decode" in f for f in m.verdict(summary)))

    def test_fail_when_cold_ttft_worse(self):
        summary = {"cold31k": {"ds4_ttft_s": 20.0, "omlx_ttft_s": 15.2}}
        self.assertTrue(any("TTFT" in f for f in m.verdict(summary)))


class _FakeArm:
    base_url = "http://127.0.0.1:18296"
    model_id = "ornith-1.5-35b-a3b"

    def __init__(self, name="ds4"):
        self.name = name

    def start(self):
        pass

    def stop(self):
        pass

    def ready(self, timeout=1):
        return True

    def stream(self, prompt, max_tokens):
        # (timing dict like sse_timings, usage dict) — cold cache, fixed rates
        return ({"prompt_tokens": 2048, "completion_tokens": 8, "ttft_s": 1.0, "decode_s": 0.1},
                {"prompt_tokens_details": {"cached_tokens": 0}})


class _CleanProc:
    pid = 1

    def __init__(self):
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        return 0

    def kill(self):
        raise AssertionError("must never SIGKILL a model process")


class _StuckProc:
    pid = 4242

    def poll(self):
        return None  # always alive

    def terminate(self):
        pass

    def wait(self, timeout):
        raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    def kill(self):
        raise AssertionError("must never SIGKILL a model process")


def _tmp_out():
    import tempfile
    return tempfile.mkdtemp(prefix="m4-ab-test-")


if __name__ == "__main__":
    unittest.main()
