"""Unit tests for the Ornith M4 A/B harness plumbing (python3 -m unittest).

No model, no GPU: the process guards, prompt building, cached-tokens assertion,
one-process-at-a-time discipline, the ds4-vs-ds4 lever mode and the gate-3
verdict are checked with fakes.
"""
import json
import os
import subprocess
import sys
import unittest
import unittest.mock

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


class OmlxRunningTest(unittest.TestCase):
    """A freshly spawned oMLX starts as .../omlx-venv/bin/omlx serve ... and may
    not have setproctitle-renamed itself to omlx-server yet; _omlx_running()
    must catch either form."""

    @staticmethod
    def _fake_run(dash_x_hit, dash_f_hit):
        def run(cmd, **kwargs):
            class _Result:
                pass
            r = _Result()
            if "-x" in cmd:
                r.stdout = "40263\n" if dash_x_hit else ""
            elif "-f" in cmd:
                r.stdout = "40263 /Users/dongnh/.local/omlx-venv/bin/omlx serve\n" if dash_f_hit else ""
            else:
                r.stdout = ""
            return r
        return run

    def test_true_when_only_dash_x_pattern_matches(self):
        with unittest.mock.patch.object(m.subprocess, "run", side_effect=self._fake_run(True, False)):
            self.assertTrue(m._omlx_running())

    def test_true_when_only_dash_f_omlx_venv_pattern_matches(self):
        with unittest.mock.patch.object(m.subprocess, "run", side_effect=self._fake_run(False, True)):
            self.assertTrue(m._omlx_running())

    def test_false_when_neither_pattern_matches(self):
        with unittest.mock.patch.object(m.subprocess, "run", side_effect=self._fake_run(False, False)):
            self.assertFalse(m._omlx_running())


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


class ProcsInEstateTest(unittest.TestCase):
    """_procs_in_estate() must ignore unrelated processes and momentary E/U
    states, and only report a model/Metal process whose E/U state persists
    across a second check ~2s later."""

    @staticmethod
    def _ps_output(lines):
        header = "  PID STAT COMMAND"
        return "\n".join([header] + lines) + "\n"

    def test_unrelated_process_in_u_is_ignored(self):
        out = self._ps_output(["92936 U   macmon"])
        with unittest.mock.patch.object(m.subprocess, "run") as run, \
             unittest.mock.patch.object(m.time, "sleep") as sleep:
            run.return_value = unittest.mock.Mock(stdout=out)
            self.assertEqual(m._procs_in_estate(), "")
        sleep.assert_not_called()

    def test_ds4_server_in_u_on_both_checks_is_reported(self):
        out = self._ps_output(["4242 U   /opt/ds4/ds4-server --metal -m /models/x.gguf"])
        with unittest.mock.patch.object(m.subprocess, "run") as run, \
             unittest.mock.patch.object(m.time, "sleep") as sleep:
            run.return_value = unittest.mock.Mock(stdout=out)
            result = m._procs_in_estate()
        self.assertIn("4242", result)
        sleep.assert_called_once()

    def test_ds4_server_in_u_on_first_check_only_is_ignored(self):
        first = self._ps_output(["4242 U   /opt/ds4/ds4-server --metal -m /models/x.gguf"])
        second = self._ps_output([])  # gone / no longer E/U
        with unittest.mock.patch.object(m.subprocess, "run") as run, \
             unittest.mock.patch.object(m.time, "sleep"):
            run.side_effect = [unittest.mock.Mock(stdout=first), unittest.mock.Mock(stdout=second)]
            self.assertEqual(m._procs_in_estate(), "")


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


class _FakeHTTPResponse:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class _RunningProc:
    def poll(self):
        return None


class _ExitedProc:
    def poll(self):
        return 1


class OmlxReadyTest(unittest.TestCase):
    """Fix round 2: the live oMLX (0.6.4) loads models lazily on the first
    request. Right after startup /api/status returns loaded_models=[]
    forever until a request comes in, so ready() must not wait on
    loaded_models; it must accept a model merely *discovered* via
    /v1/models, then force-load it itself via one unmeasured request."""

    @staticmethod
    def _status_body(status="ok"):
        return json.dumps({"status": status, "models_loaded": 0, "loaded_models": []}).encode()

    @staticmethod
    def _models_body(discovered):
        data = [{"id": m.OMLX_MODEL}] if discovered else []
        return json.dumps({"data": data}).encode()

    def test_ready_true_when_status_ok_and_model_discovered_not_loaded(self):
        arm = m.OmlxArm(_tmp_out())
        arm.proc = _RunningProc()
        arm.stream = unittest.mock.Mock(return_value=({}, {}))

        def fake_urlopen(url, timeout=2):
            if "/api/status" in url:
                return _FakeHTTPResponse(self._status_body())
            return _FakeHTTPResponse(self._models_body(True))

        with unittest.mock.patch.object(m.urllib.request, "urlopen", side_effect=fake_urlopen):
            self.assertTrue(arm.ready(timeout=3))
        arm.stream.assert_called_once()  # the unmeasured load request

    def test_ready_keeps_polling_while_v1_models_lacks_the_model(self):
        arm = m.OmlxArm(_tmp_out())
        arm.proc = _RunningProc()
        arm.stream = unittest.mock.Mock(return_value=({}, {}))
        seen = {"n": 0}

        def fake_urlopen(url, timeout=2):
            if "/api/status" in url:
                return _FakeHTTPResponse(self._status_body())
            seen["n"] += 1
            return _FakeHTTPResponse(self._models_body(seen["n"] >= 3))

        with unittest.mock.patch.object(m.urllib.request, "urlopen", side_effect=fake_urlopen), \
             unittest.mock.patch.object(m.time, "sleep", lambda s: None):
            self.assertTrue(arm.ready(timeout=10))
        self.assertGreaterEqual(seen["n"], 3)

    def test_ready_raises_systemexit_when_process_exits(self):
        arm = m.OmlxArm(_tmp_out())
        arm.proc = _ExitedProc()
        with self.assertRaises(SystemExit):
            arm.ready(timeout=5)


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


class SignalExitTest(unittest.TestCase):
    """Important 2 (final review): SIGTERM/SIGHUP must not orphan the model
    arm. _raise_signal_exit turns the signal into a SystemExit so
    interleave()'s `finally: arm.stop()` runs."""

    def test_raise_signal_exit_raises_systemexit_with_128_plus_signum(self):
        with self.assertRaises(SystemExit) as cm:
            m._raise_signal_exit(15, None)  # SIGTERM = 15
        self.assertEqual(cm.exception.code, 143)

    def test_sigterm_during_interleave_still_stops_the_arm(self):
        stopped = []

        class _SigtermArm(_FakeArm):
            def ready(self, timeout=1):
                # Simulate a SIGTERM landing mid-measurement: the installed
                # handler raises SystemExit(143), which is what a real
                # signal delivery would do at any point in the try block.
                m._raise_signal_exit(15, None)

            def stop(self):
                stopped.append(self.name)

        with self.assertRaises(SystemExit):
            m.interleave({"ds4": lambda: _SigtermArm()}, contexts=[2048], cold_tokens=0,
                        filler="x" * 100000, max_tokens=8, warmup=0, order=("ds4",),
                        guard=lambda: None, wait_free=lambda: None,
                        swap_used=iter([0.0] * 8).__next__, wait_idle=lambda: None)
        self.assertEqual(stopped, ["ds4"])


class PairedModeTest(unittest.TestCase):
    def test_paired_nonces_distinct_per_rep_and_block_shared_across_arms(self):
        a = {m.paired_nonce(b, c, r) for b in (0, 1) for c in ("2048", "32768") for r in (0, 1)}
        self.assertEqual(len(a), 8)
        self.assertEqual(m.paired_nonce(0, "2048", 1), m.paired_nonce(0, "2048", 1))

    def test_text_sha_separates_reasoning_and_content(self):
        def ev(parts):
            return [(0.0, {"choices": [{"delta": {k: v}}]}) for k, v in parts]
        one = m.stream_text_sha(ev([("reasoning_content", "ab"), ("content", "c")]))
        two = m.stream_text_sha(ev([("reasoning_content", "a"), ("content", "bc")]))
        self.assertNotEqual(one, two)
        r1 = m.stream_text_sha(ev([("reasoning_content", "x")]))
        r2 = m.stream_text_sha(ev([("reasoning_content", "y")]))
        self.assertNotEqual(r1, r2)
        # chunking does not matter, only the concatenated text
        self.assertEqual(m.stream_text_sha(ev([("content", "ab")])),
                         m.stream_text_sha(ev([("content", "a"), ("content", "b")])))

    def _run(self, role, block, samples):
        return {"arm": role, "block": block, "rows": {},
                "samples": [{"ctx": c, "rep": r, "decode_tps": t, "text_sha": s} for c, r, t, s in samples]}

    def test_paired_summary_gain_and_spread(self):
        runs = {"base": [self._run("base", 0, [("2048", 0, 100.0, "h")]),
                         self._run("base", 0, [("2048", 0, 100.0, "h")])],
                "lever": [self._run("lever", 0, [("2048", 0, 104.0, "h")]),
                          self._run("lever", 0, [("2048", 0, 106.0, "h")])]}
        runs["base"].append(self._run("base", 1, [("2048", 0, 90.0, "k")]))
        runs["lever"].append(self._run("lever", 1, [("2048", 0, 90.0, "k")]))
        p = m.paired_summary(runs, ["2048"])["2048"]
        self.assertEqual(p["n"], 2)
        self.assertAlmostEqual(p["mean"], 0.025)          # (+5% + 0%) / 2
        self.assertAlmostEqual(p["min"], 0.0)
        self.assertAlmostEqual(p["max"], 0.05)
        self.assertEqual(p["mismatches"], 0)
        self.assertGreater(p["sd"], 0.0)

    def test_paired_summary_all_mismatch(self):
        runs = {"base": [self._run("base", 0, [("2048", 0, 100.0, "h")])],
                "lever": [self._run("lever", 0, [("2048", 0, 80.0, "DIFFERENT")])]}
        p = m.paired_summary(runs, ["2048"])["2048"]
        self.assertEqual(p["n"], 0)
        self.assertIsNone(p["mean"])
        self.assertEqual(p["mismatches"], 1)
        self.assertEqual(p["mismatch_samples"], [{"block": 0, "rep": 0}])
        self.assertIn("n=0", m.format_paired("2048", p))

    def test_interleave_paired_blocks_send_same_prompts_to_both_arms(self):
        prompts = []

        class _Rec(_FakeArm):
            def stream(self, prompt, max_tokens):
                if not prompt.startswith(("nonce-warm", "nonce-cal")):
                    prompts.append((self.name, prompt[:24]))
                t, u = _FakeArm.stream(self, prompt, max_tokens)
                return {**t, "text_sha": "same"}, u

        s = m.interleave({"base": lambda: _Rec(name="base"), "lever": lambda: _Rec(name="lever")},
                         contexts=[2048], cold_tokens=0, filler="x" * 100000, max_tokens=8, warmup=0,
                         order=m.LEVER_ORDER, guard=lambda: None, wait_free=lambda: None,
                         swap_used=iter([0.0] * 32).__next__, wait_idle=lambda: None,
                         blocks=2, paired=True, reps=2)
        self.assertEqual(len(prompts), 2 * 4 * 2)                  # blocks x runs x reps
        b0 = {p for _, p in prompts[:8]}
        b1 = {p for _, p in prompts[8:]}
        self.assertEqual(len(b0), 2)                                # 2 reps, shared by all 4 runs
        self.assertEqual(len(b1), 2)
        self.assertFalse(b0 & b1)                                   # fresh nonces per block
        self.assertEqual(s["paired"]["2048"]["n"], 4)               # 2 blocks x 2 reps
        self.assertEqual(s["paired"]["2048"]["mismatches"], 0)

    def test_reps_average_into_rows(self):
        rates = iter([100.0, 110.0])

        class _Two(_FakeArm):
            def stream(self, prompt, max_tokens):
                t, u = _FakeArm.stream(self, prompt, max_tokens)
                if prompt.startswith(("nonce-warm", "nonce-cal")):
                    return t, u
                return {**t, "decode_s": 7 / next(rates)}, u      # request_rates: (8 - 1) / decode_s

        run = m.measure_arm(_Two(name="base"), [2048], "x" * 100000, 8, warmup=0,
                            swap_used=iter([0.0, 0.0]).__next__, wait_idle=lambda: None, reps=2)
        self.assertEqual(len(run["samples"]), 2)
        self.assertAlmostEqual(run["rows"]["2048"]["decode_tps"], 105.0, places=6)

    def test_paired_flags_rejected_in_baseline_mode(self):
        with unittest.mock.patch.object(sys, "argv", ["m4_ab.py", "--mode", "baseline", "--ds4-model", "x",
                                                      "--out", _tmp_out(), "--paired"]):
            import io
            err = io.StringIO()
            with unittest.mock.patch.object(sys, "stderr", err), self.assertRaises(SystemExit) as cm:
                m.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("apply to --mode lever only", err.getvalue())


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
