import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_suites  # noqa: E402
import graders  # noqa: E402
from test_graders import MBPP_ROW  # noqa: E402

SPEED = {"prefill_s": 0.1, "prefill_tps": 100.0, "think_tokens": 10, "gen_tokens": 20, "finish": "stop",
         "total_s": 1.1, "decode_tps": 40.0}


class FakeSampler:
    def start(self):
        pass

    def stop(self):
        return {"peak_wired_gib": 50.0, "swapouts": 0}


class FakeCtx:
    def __init__(self, data_dir, answer, ctx_limit=262144):
        self.data_dir, self.answer, self.ctx_limit = pathlib.Path(data_dir), answer, ctx_limit
        self.root = eval_suites.server.ROOT
        self.base_url = "http://fake"
        self.sampler_factory = FakeSampler
        self.prompts = []

    def ask(self, prompt_or_messages, max_tokens=16384, extra=None):
        p = prompt_or_messages if isinstance(prompt_or_messages, str) else prompt_or_messages[-1]["content"]
        self.prompts.append((p, max_tokens, extra))
        return dict(SPEED, content=self.answer(p), reasoning="", usage={"prompt_tokens": len(p) // 3},
                    seconds=1.0)

    def take_log(self):
        return dict(SPEED)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


class RealCtx(unittest.TestCase):
    def test_ask_attributes_log_lines_and_prefers_http_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            log = pathlib.Path(d) / "server.log"
            # A late line from the previous request: it must not be attributed to this one.
            log.write_text("ds4-server: chat ctx=0..9:9 thinking closed after 999 tokens\n")

            def fake_chat(base_url, messages, max_tokens=16384, extra=None, timeout=7200):
                with open(log, "a") as f:
                    f.write("ds4-server: chat ctx=0..81:81 prompt done 0.500s\n"
                            "ds4-server: chat ctx=0..81:81 prefill chunk 81/81 (100.0%) chunk=0.00 t/s "
                            "avg=162.00 t/s 0.500s\n"
                            "ds4-server: chat ctx=81..131:50 gen=50 decoding chunk=40.00 t/s avg=40.00 t/s 1.2s\n")
                return {"content": "x", "reasoning": "", "usage": {"completion_tokens": 77},
                        "finish_reason": "length", "seconds": 2.0}

            ctx = eval_suites.Ctx("http://fake", eval_suites.server.LogCursor(log), d, 262144, d)
            with mock.patch.object(eval_suites.server, "chat", fake_chat):
                r = ctx.ask("hi")
        self.assertEqual((r["gen_tokens"], r["finish"]), (77, "length"))
        self.assertEqual((r["prefill_s"], r["prefill_tps"], r["decode_tps"]), (0.5, 162.0, 40.0))
        self.assertIsNone(r["think_tokens"])


class Suites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_code(self):
        write_jsonl(self.data / "mbpp.jsonl", [MBPP_ROW])
        # HumanEval prompts get code that crashes at import, so no HE row can pass by accident.
        answer = lambda p: ("```python\n" + MBPP_ROW["code"] + "```" if "shared elements" in p
                            else "```python\nraise RuntimeError('fake')\n```")
        rows = eval_suites.run_code(FakeCtx(self.data, answer))
        he = graders.bench_module("humaneval_mini").PROBLEMS
        self.assertEqual(len(rows), len(he) + 1)
        self.assertTrue(rows[-1]["passed"])
        self.assertFalse(any(r["passed"] for r in rows[:-1]))
        self.assertTrue(all(r["suite"] == "code" and r["seconds"] == 1.0 for r in rows))

    def test_ifeval(self):
        write_jsonl(self.data / "ifeval.jsonl", [{"key": 7, "prompt": "p", "kwargs": [{}, {}],
                                                   "instruction_id_list": ["punctuation:no_comma",
                                                                           "change_case:english_lowercase"]}])
        rows = eval_suites.run_ifeval(FakeCtx(self.data, lambda p: "all lower no comma"))
        self.assertEqual(rows[0]["id"], "7")
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(rows[0]["instructions"], [["punctuation:no_comma", True],
                                                   ["change_case:english_lowercase", True]])

    def test_vi(self):
        rows = eval_suites.run_vi(FakeCtx(self.data, lambda p: "Đáp án: Hà Nội. 你好"))
        by = {}
        for r in rows:
            by.setdefault(r["suite"], []).append(r)
        self.assertEqual(len(by["vi_knowledge"]), 30)
        self.assertEqual(len(by["vi_writing"]), 10)
        self.assertEqual(len(by["vi_speed"]), 5)
        self.assertTrue(by["vi_knowledge"][0]["passed"])
        self.assertIsNone(by["vi_writing"][0]["passed"])
        self.assertTrue(all(r["cjk"] == 2 for r in rows))

    def test_uncensor(self):
        write_jsonl(self.data / "harmful.jsonl", [{"text": "bad one"}, {"text": "bad two"}])
        write_jsonl(self.data / "harmless.jsonl", [{"text": "good one"}, {"text": "good two"}])
        answer = lambda p: "I'm sorry, but I can't help with that." if p.startswith("bad") else "Sure: done."
        rows = eval_suites.run_uncensor(FakeCtx(self.data, answer))
        harmful = [r for r in rows if r["suite"] == "uncensor_harmful"]
        harmless = [r for r in rows if r["suite"] == "uncensor_harmless"]
        self.assertTrue(all(r["refused"] and not r["passed"] for r in harmful))
        self.assertTrue(all(not r["refused"] and r["passed"] for r in harmless))

    def _haystack(self):
        # A stand-in for the frozen ds4.c snapshot, long enough for the 240k tier (727,200 characters).
        (self.data / eval_suites.HAYSTACK).write_text("int x = 1;\n" * 80000)

    def test_longctx_needs_the_frozen_haystack(self):
        with self.assertRaises(RuntimeError) as cm:
            eval_suites.run_longctx(FakeCtx(self.data, lambda p: "7314-QX", ctx_limit=130000))
        self.assertIn("fetch_data.py", str(cm.exception))

    def test_longctx_skips_tiers_beyond_ctx(self):
        self._haystack()
        ctx = FakeCtx(self.data, lambda p: "7314-QX", ctx_limit=130000)
        rows = eval_suites.run_longctx(ctx)
        needles = {r["id"]: r for r in rows if r["suite"] == "longctx"}
        self.assertTrue(needles["needle-120k"]["passed"])
        for tier in ("240k", "480k", "960k"):
            self.assertIsNone(needles["needle-" + tier]["passed"])
            self.assertIn("skipped", needles["needle-" + tier])
        self.assertEqual(len(ctx.prompts), 1)
        self.assertLess(len(ctx.prompts[0][0]) / eval_suites.CHARS_PER_TOKEN, 130000)
        self.assertEqual(ctx.prompts[0][2], {"chat_template_kwargs": {"enable_thinking": False}})
        mem = [r for r in rows if r["suite"] == "longctx_mem"]
        self.assertEqual(mem[0]["peak_wired_gib"], 50.0)

    def test_longctx_doc_questions_on_240k(self):
        self._haystack()
        def answer(p):
            tail = p[-300:]
            for question, value in eval_suites.DOC_QA:
                if question in tail:
                    return value
            return "7314-QX"
        rows = eval_suites.run_longctx(FakeCtx(self.data, answer, ctx_limit=262144))
        ids = {r["id"]: r["passed"] for r in rows if r["suite"] == "longctx"}
        self.assertEqual(ids["docqa-0"], True)
        self.assertEqual(ids["docqa-2"], True)
        self.assertEqual(ids["needle-240k"], True)

    def test_haystack_puts_needle_mid_document(self):
        self._haystack()
        doc = eval_suites.haystack(self.data / eval_suites.HAYSTACK, 10000)
        pos = doc.index(eval_suites.NEEDLE)
        self.assertTrue(0.4 < pos / len(doc) < 0.6)

    def _fake_gateway(self):
        """Fake gateway harness + suites modules that behave like the real ones against ds4-server:
        /mcp answers 404, and each suite's run() calls client.chat() the way the real suite does."""
        class Case:
            def __init__(self, cid, inp):
                self.id, self.inp, self.suite = cid, inp, ""

        class Result:
            def __init__(self, passed):
                self.passed, self.score, self.metrics = passed, 1.0 if passed else 0.0, {"m": 1}

        chats = []

        class ArmClient:
            def __init__(self, base, key, model):
                self.base = base

            def post(self, path, body, timeout=300):
                raise OSError("HTTP Error 404: unknown endpoint %s" % path)

            def chat(self, prompt, max_tokens=800, chat_template_kwargs=None, **kw):
                chats.append({"max_tokens": max_tokens, "chat_template_kwargs": chat_template_kwargs})
                raw = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                       "usage": {"completion_tokens": 9}}
                return "ok", 0.5, raw

        class ToolCall:
            name = "toolcall"

            def available(self, client):
                r, _ = client.post("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, timeout=30)
                return bool(r["result"]["tools"]), ""

            def cases(self):
                return [Case("sel-ok", {"kind": "pos"}), Case("neg-bad", {"kind": "neg"}),
                        Case("xfer-ok", {"kind": "xfer"})]

            def run(self, client, case):
                txt, dt, raw = client.chat("task", max_tokens=600,
                                           chat_template_kwargs={"enable_thinking": False})
                return {"text": txt, "raw": raw, "latency": dt}

            def grade(self, case, output, judge_client):
                assert judge_client is None
                return Result(case.id.endswith("ok"))

        class Faithfulness:
            name = "faithfulness"

            def cases(self):
                return [Case("judge", {"no_fabricate": "x"}), Case("fact-ok", {"q": "q"})]

            def run(self, client, case):
                content, dt, _ = client.chat(case.inp["q"], max_tokens=500)
                return {"answer": content, "latency": dt}

            def grade(self, case, output, judge_client):
                assert judge_client is None
                return Result(case.id.endswith("ok"))

        harness = types.ModuleType("harness")
        harness.Client = ArmClient
        suites = types.ModuleType("suites")
        suites.BY_NAME = {"toolcall": ToolCall(), "faithfulness": Faithfulness()}
        return {"harness": harness, "suites": suites}, chats

    def test_tools_adapter(self):
        catalog = {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "read_file"}]}}
        (self.data / "mcp_tools.json").write_text(json.dumps(catalog))
        modules, chats = self._fake_gateway()
        with mock.patch.dict(sys.modules, modules):
            rows = eval_suites.run_tools(FakeCtx(self.data, lambda p: ""))
        self.assertEqual([r["suite"] for r in rows], ["tools_pos", "tools_neg", "tools_xfer", "faithfulness"])
        self.assertEqual([r["passed"] for r in rows], [True, False, True, True])
        self.assertTrue(all(r["seconds"] == 0.5 for r in rows))
        self.assertEqual([r["gen_tokens"] for r in rows[:3]], [9, 9, 9])
        self.assertEqual(rows[0]["finish"], "stop")
        # faithfulness does not set thinking itself; straight to ds4-server it would think inside
        # its 500-token budget, so the harness pins thinking off like toolcall does.
        self.assertEqual(chats[-1], {"max_tokens": 500, "chat_template_kwargs": {"enable_thinking": False}})

    def test_tools_without_frozen_catalog_fails_clearly(self):
        modules, _ = self._fake_gateway()
        with mock.patch.dict(sys.modules, modules), self.assertRaises(RuntimeError) as cm:
            eval_suites.run_tools(FakeCtx(self.data, lambda p: ""))
        self.assertIn("fetch_data.py", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
