"""The server-backed suites. Each run_<suite>(ctx) returns a list of row dicts."""
import json
import os
import pathlib
import sys

import graders
import ifeval_checks
import server

HERE = pathlib.Path(__file__).resolve().parent
# The five Vietnamese prompts of speed-bench/think-budget/measure.py, kept identical so the PROD receipt
# (speed-bench/think-budget/runs/budget-0.json) can check this harness.
THINK_BUDGET_VI = [
    "Giải thích ngắn gọn cách bộ nhớ đệm KV giúp mô hình ngôn ngữ sinh văn bản nhanh hơn.",
    "Viết một đoạn văn khoảng 120 chữ giới thiệu vịnh Hạ Long cho khách du lịch.",
    "So sánh ưu và nhược điểm của Python và Go khi viết dịch vụ web.",
    "Tóm tắt ý chính của định luật Ohm và cho một ví dụ tính toán.",
    "Một cửa hàng giảm giá 20% rồi giảm thêm 10% trên giá đã giảm. Tổng cộng giảm bao nhiêu phần trăm?",
]
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
NEEDLE_VALUE = "7314-QX"
NEEDLE = "\n/* NOTE: The vault code for project Heliotrope is 7314-QX. */\n"
NEEDLE_Q = "What is the vault code for project Heliotrope? Reply with the code only."
DOC_QA = [
    ("In the C source above, what integer value is DS4_MAX_PLE_HEADS set to? Reply with the number only.", "64"),
    ("In the C source above, what is the value of DS4_DEFAULT_ROPE_YARN_BETA_FAST? Reply with the number only.",
     "32"),
    ("In the C source above, what size in bytes does the static assert give for block_iq2_xxs? "
     "Reply with the number only.", "66"),
]
# The 128K / 256K / 512K / 1M tiers, with margin for the tokenizer ratio's spread.
TIERS = [("120k", 120000), ("240k", 240000), ("480k", 480000), ("960k", 960000)]
CHARS_PER_TOKEN = 3.03  # measured 2026-09-23: 685K characters of ds4.c = 226,292 tokens
CTX_MARGIN = 8192
SPEED_KEYS = ("seconds", "prefill_s", "prefill_tps", "think_tokens", "gen_tokens", "decode_tps", "finish",
              "total_s")


class Ctx:
    """What a suite needs from a running server."""

    def __init__(self, base_url, cursor, data_dir, ctx_limit, root):
        self.base_url, self.cursor = base_url, cursor
        self.data_dir, self.ctx_limit, self.root = pathlib.Path(data_dir), ctx_limit, pathlib.Path(root)
        self.sampler_factory = server.MemSampler

    def take_log(self):
        return server.parse_request_log(self.cursor.take())

    def ask(self, prompt_or_messages, max_tokens=16384, extra=None):
        messages = ([{"role": "user", "content": prompt_or_messages}]
                    if isinstance(prompt_or_messages, str) else prompt_or_messages)
        self.cursor.take()  # drop log lines from before this request
        out = server.chat(self.base_url, messages, max_tokens, extra)
        out.update(self.take_log())
        out.update(_http_numbers(out.get("usage"), out.get("finish_reason")))
        return out


def _http_numbers(usage, finish_reason):
    """Generated tokens and finish reason from the HTTP response. ds4-server logs its finish line only
    after sending the response, so the log copy can be late; the response is authoritative."""
    out = {}
    if (usage or {}).get("completion_tokens") is not None:
        out["gen_tokens"] = usage["completion_tokens"]
    if finish_reason:
        out["finish"] = finish_reason
    return out


def _speed(r):
    return {k: r.get(k) for k in SPEED_KEYS}


def _load_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def run_code(ctx):
    he = graders.bench_module("humaneval_mini")
    for problem in he.PROBLEMS:
        r = ctx.ask(he.build_prompt(problem))
        ok, detail = graders.grade_code(problem, r["content"])
        yield (dict(_speed(r), suite="code", id=problem["task_id"], passed=ok, detail=detail))
    for row in _load_jsonl(ctx.data_dir / "mbpp.jsonl"):
        problem = graders.mbpp_problem(row)
        r = ctx.ask(graders.mbpp_prompt(row))
        ok, detail = graders.grade_code(problem, r["content"])
        yield (dict(_speed(r), suite="code", id=problem["task_id"], passed=ok, detail=detail))


def run_ifeval(ctx):
    for row in _load_jsonl(ctx.data_dir / "ifeval.jsonl"):
        r = ctx.ask(row["prompt"])
        ok, each = ifeval_checks.check_prompt(row, r["content"])
        yield (dict(_speed(r), suite="ifeval", id=str(row["key"]), passed=ok,
                         instructions=[[i, e] for i, e in zip(row["instruction_id_list"], each)]))


def run_vi(ctx):
    for item in json.loads((HERE / "data" / "vi_knowledge.json").read_text()):
        r = ctx.ask(item["q"] + " Trả lời ngắn gọn.")
        yield (dict(_speed(r), suite="vi_knowledge", id=item["id"],
                         passed=graders.keyword_groups_hit(r["content"], item["groups"]),
                         cjk=graders.cjk_count(r["content"]), answer=r["content"]))
    for item in json.loads((HERE / "data" / "vi_writing.json").read_text()):
        r = ctx.ask(item["prompt"])
        yield (dict(_speed(r), suite="vi_writing", id=item["id"], passed=None,
                         cjk=graders.cjk_count(r["content"]), answer=r["content"]))
    for i, prompt in enumerate(THINK_BUDGET_VI):
        r = ctx.ask(prompt)
        yield (dict(_speed(r), suite="vi_speed", id="tb-vi-%d" % i, passed=None,
                         cjk=graders.cjk_count(r["content"]), answer=r["content"]))


def run_uncensor(ctx):
    for kind in ("harmful", "harmless"):
        for i, row in enumerate(_load_jsonl(ctx.data_dir / ("%s.jsonl" % kind))):
            r = ctx.ask(row["text"])
            refused = graders.is_refusal(r["content"])
            yield (dict(_speed(r), suite="uncensor_" + kind, id="%s-%02d" % (kind, i),
                             passed=not refused, refused=refused, answer=r["content"][:600]))


MCP_CATALOG = "mcp_tools.json"


class _ArmClient:
    """The gateway eval client pointed at the arm.

    - /mcp tools/list comes from the catalog fetch_data.py froze: ds4-server has no /mcp, and a frozen
      catalog keeps the graded tools identical across arms run days apart.
    - chat() pins thinking off unless the suite sets it (toolcall does; faithfulness does not, and
      straight to ds4-server it would think inside its 500-token budget)."""

    def __init__(self, client, catalog):
        self._client, self._catalog = client, catalog

    def chat(self, *args, **kwargs):
        if not kwargs.get("chat_template_kwargs"):
            kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        return self._client.chat(*args, **kwargs)

    def post(self, path, body, timeout=300):
        if path == "/mcp" and body.get("method") == "tools/list":
            return self._catalog, 0.0
        raise RuntimeError("%s is a gateway endpoint; ds4-server does not serve it" % path)


def _tool_row(ctx, out, **fields):
    raw = out.get("raw") or {}
    choice = (raw.get("choices") or [{}])[0]
    speed = dict(ctx.take_log(), seconds=round(out["latency"], 2))
    speed.update(_http_numbers(raw.get("usage"), choice.get("finish_reason")))
    return dict(_speed(speed), **fields)


def run_tools(ctx):
    """The gateway's toolcall suite and the code-graded faithfulness cases, pointed at this server."""
    catalog_path = ctx.data_dir / MCP_CATALOG
    if not catalog_path.exists():
        raise RuntimeError("%s missing: run fetch_data.py while the gateway stack is up" % catalog_path)
    os.environ["EVAL_CONCURRENCY"] = "1"
    gateway_evals = str(graders.GATEWAY_REPO / "evals")
    if gateway_evals not in sys.path:
        sys.path.append(gateway_evals)
    import harness
    from suites import BY_NAME
    client = _ArmClient(harness.Client(base=ctx.base_url, key="nextgen-eval", model="ds4"),
                        json.loads(catalog_path.read_text()))
    toolcall = BY_NAME["toolcall"]
    ok, note = toolcall.available(client)
    if not ok:
        raise RuntimeError("gateway toolcall suite unavailable: %s" % note)
    for case in toolcall.cases():
        case.suite = toolcall.name
        ctx.take_log()
        out = toolcall.run(client, case)
        res = toolcall.grade(case, out, None)
        yield (_tool_row(ctx, out, suite="tools_" + case.inp["kind"], id=case.id, passed=bool(res.passed),
                              score=res.score, metrics=res.metrics))
    faith = BY_NAME["faithfulness"]
    for case in faith.cases():
        if "no_fabricate" in case.inp:
            continue  # judge-graded: the judge would be the model under test, a different one per arm
        case.suite = faith.name
        ctx.take_log()
        out = faith.run(client, case)
        res = faith.grade(case, out, None)
        yield (_tool_row(ctx, out, suite="faithfulness", id=case.id, passed=bool(res.passed),
                              score=res.score, metrics=res.metrics))


HAYSTACK = "haystack.c"  # ds4.c frozen by fetch_data.py: later sub-projects edit ds4.c on this branch


def haystack(path, tokens, depth=0.5):
    """The first tokens*CHARS_PER_TOKEN characters of the frozen ds4.c, with the needle at `depth`."""
    src = pathlib.Path(path).read_text(errors="replace")
    n = int(tokens * CHARS_PER_TOKEN)
    if n > len(src):
        raise ValueError("%s has %d characters, need %d" % (path, len(src), n))
    text = src[:n]
    cut = text.rfind("\n", 0, int(n * depth)) + 1
    return text[:cut] + NEEDLE + text[cut:]


def run_longctx(ctx):
    source = ctx.data_dir / HAYSTACK
    if not source.exists():
        raise RuntimeError("%s missing: run fetch_data.py" % source)
    sampler = ctx.sampler_factory()
    sampler.start()
    try:
        for label, tokens in TIERS:
            if tokens + CTX_MARGIN > ctx.ctx_limit:
                yield ({"suite": "longctx", "id": "needle-" + label, "passed": None,
                             "skipped": "ctx limit %d" % ctx.ctx_limit})
                continue
            doc = "Here is a C source file.\n\n" + haystack(source, tokens) + "\n\n"
            r = ctx.ask(doc + NEEDLE_Q, max_tokens=512, extra=NO_THINK)
            yield (dict(_speed(r), suite="longctx", id="needle-" + label,
                             passed=graders.needle_hit(r["content"], NEEDLE_VALUE),
                             prompt_tokens=r["usage"].get("prompt_tokens"), answer=r["content"][:200]))
            if label == "240k":
                for j, (question, value) in enumerate(DOC_QA):
                    r = ctx.ask(doc + question, max_tokens=512, extra=NO_THINK)
                    yield (dict(_speed(r), suite="longctx", id="docqa-%d" % j,
                                     passed=graders.keyword_hit(r["content"], value),
                                     prompt_tokens=r["usage"].get("prompt_tokens"), answer=r["content"][:200]))
    finally:
        mem = sampler.stop()
    yield (dict(mem, suite="longctx_mem", id="memory", passed=None))


SERVER_SUITES = {"code": run_code, "ifeval": run_ifeval, "vi": run_vi, "uncensor": run_uncensor,
                 "tools": run_tools, "longctx": run_longctx}
