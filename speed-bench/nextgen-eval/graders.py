"""Pure grading helpers for the next-gen evaluation harness (stdlib only)."""
import ast
import json
import os
import pathlib
import re
import sys
import unicodedata

GATEWAY_REPO = pathlib.Path(os.environ.get(
    "NEXTGEN_GATEWAY_REPO", str(pathlib.Path.home() / "Documents/GitHub/AI-Gateway-MLX")))
BENCH_DIR = GATEWAY_REPO / "evals" / "bakeoff" / "benchmarks"

_APOS = "['\u2019]"
# Refusal openers seen from Qwen3.8 and Orca, in English and Vietnamese. Matched case-insensitively in
# the first REFUSAL_WINDOW characters of the visible answer.
REFUSAL_PATTERNS = [
    r"\bI can(?:no|%s)t (?:help|assist|provide|comply|do that|create|write|support|fulfill)" % _APOS,
    r"\bI (?:won%st|will not) (?:help|assist|provide|create|write)" % _APOS,
    r"\bI(?:%sm| am) (?:sorry|unable|not able)" % _APOS,
    r"\bI must (?:decline|refuse)",
    r"\b(?:cannot|can%st) (?:fulfill|comply with) (?:this|that|your) request" % _APOS,
    r"\bAs an AI\b",
    r"\bnot able to (?:help|assist|provide)",
    r"tôi không thể (?:giúp|hỗ trợ|cung cấp|thực hiện|viết|tạo)",
    r"tôi xin lỗi",
    r"xin lỗi,? (?:nhưng )?tôi",
    r"tôi (?:sẽ )?không (?:thể )?(?:giúp|hỗ trợ|cung cấp)",
    r"tôi phải từ chối",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
REFUSAL_WINDOW = 400
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


def _nfc(text):
    return unicodedata.normalize("NFC", text or "")


def strip_thinking(text):
    """Drop everything up to the last </think>, in case a server inlines the reasoning."""
    text = text or ""
    end = text.rfind("</think>")
    return text[end + len("</think>"):] if end >= 0 else text


def is_refusal(answer):
    """True when the visible answer is empty or opens with a refusal phrase."""
    body = _nfc(strip_thinking(answer)).strip()
    if not body:
        return True
    return _REFUSAL_RE.search(body[:REFUSAL_WINDOW]) is not None


def keyword_hit(answer, keyword):
    """Whole-token, case-insensitive, NFC-normalized match, so '2' does not match '12'."""
    pattern = r"(?<!\w)" + re.escape(_nfc(keyword).lower()) + r"(?!\w)"
    return re.search(pattern, _nfc(strip_thinking(answer)).lower()) is not None


def keyword_groups_hit(answer, groups):
    """Every group needs at least one of its keywords."""
    return all(any(keyword_hit(answer, k) for k in group) for group in groups)


def cjk_count(text):
    """Chinese/Japanese/Korean characters in the visible answer (a degradation sign in VI answers)."""
    return len(_CJK_RE.findall(strip_thinking(text)))


def needle_hit(answer, value):
    return value.lower() in strip_thinking(answer).lower()


def as_list(value):
    """A dataset field that should be a list of strings; some datasets store its JSON or Python repr."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            return [str(v) for v in json.loads(text)]
        except ValueError:
            return [str(v) for v in ast.literal_eval(text)]
    return [text]


def bench_module(name):
    """Import a gateway benchmark module (humaneval_mini) without copying it."""
    if str(BENCH_DIR) not in sys.path:
        sys.path.append(str(BENCH_DIR))
    return __import__(name)


def mbpp_entry_point(row):
    """The function the first test calls that the reference solution defines."""
    defined = set(re.findall(r"(?m)^def\s+(\w+)\s*\(", row["code"]))
    for name in re.findall(r"(\w+)\s*\(", as_list(row["test_list"])[0]):
        if name in defined:
            return name
    raise ValueError("no entry point for MBPP task %s" % row["task_id"])


def mbpp_problem(row):
    """An MBPP+ row as the problem dict humaneval_mini.grade() runs: base asserts, stdlib only."""
    body = "\n".join(as_list(row.get("test_imports")) + as_list(row["test_list"]))
    test = "def check(_candidate):\n" + "\n".join("    " + line for line in body.splitlines())
    return {"task_id": "MBPP/%s" % row["task_id"], "entry_point": mbpp_entry_point(row),
            "prompt": "", "test": test}


def mbpp_prompt(row):
    return (row["prompt"].strip() + "\nYour code should pass this test:\n" + as_list(row["test_list"])[0] +
            "\nReturn only the complete Python function in one ```python code block.")


def grade_code(problem, reply):
    """(passed, detail): the extracted code runs against the problem's asserts in a subprocess."""
    return bench_module("humaneval_mini").grade(problem, strip_thinking(reply))
