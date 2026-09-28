"""Strict checks for the IFEval instruction ids this harness supports, after google-research's
instruction_following_eval. Words and sentences are counted with regexes instead of nltk, so absolute
scores differ from published IFEval numbers; both arms of a comparison use the same checks."""
import collections
import json
import re


def _count_words(text):
    return len(re.findall(r"\w+", text))


def _count_sentences(text):
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def _compare(actual, relation, target):
    if relation == "less than":
        return actual < target
    if relation == "at least":
        return actual >= target
    raise ValueError("unknown relation %r" % relation)


def _no_comma(r, kw):
    return re.search(r",", r) is None


def _number_words(r, kw):
    return _compare(_count_words(r), kw["relation"], kw["num_words"])


def _number_sentences(r, kw):
    return _compare(_count_sentences(r), kw["relation"], kw["num_sentences"])


def _forbidden_words(r, kw):
    return not any(re.search(r"\b" + re.escape(w) + r"\b", r, re.IGNORECASE) for w in kw["forbidden_words"])


def _highlighted_sections(r, kw):
    n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", r) if h.strip("*").strip())
    n += sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", r) if h[2:-2].strip())
    return n >= kw["num_highlights"]


def _frequency(r, kw):
    count = len(re.findall(re.escape(kw["keyword"]), r, re.IGNORECASE))
    return _compare(count, kw["relation"], kw["frequency"])


def _quotation(r, kw):
    v = r.strip()
    return len(v) > 1 and v[0] == '"' and v[-1] == '"'


def _lowercase(r, kw):
    return r.islower()


def _existence(r, kw):
    return all(re.search(re.escape(k), r, re.IGNORECASE) for k in kw["keywords"])


def _title(r, kw):
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", r))


def _letter_frequency(r, kw):
    count = collections.Counter(r.lower())[kw["letter"].lower()]
    return _compare(count, kw["let_relation"], kw["let_frequency"])


def _bullet_lists(r, kw):
    stars = re.findall(r"^\s*\*[^\*].*$", r, re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", r, re.MULTILINE)
    return len(stars) + len(dashes) == kw["num_bullets"]


def _placeholders(r, kw):
    return len(re.findall(r"\[.*?\]", r)) >= kw["num_placeholders"]


def _paragraphs(r, kw):
    parts = re.split(r"\s?\*\*\*\s?", r)
    count = len(parts)
    for i, part in enumerate(parts):
        if not part.strip():
            if i == 0 or i == len(parts) - 1:
                count -= 1
            else:
                return False
    return count == kw["num_paragraphs"]


def _end_checker(r, kw):
    return r.strip().strip('"').lower().endswith(kw["end_phrase"].strip().lower())


def _postscript(r, kw):
    marker = kw["postscript_marker"].lower()
    if marker == "p.p.s":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "p.s.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(marker) + r".*$"
    return bool(re.findall(pattern, r.lower(), re.MULTILINE))


def _capital(r, kw):
    return r.isupper()


def _capital_word_frequency(r, kw):
    words = re.findall(r"\b[\w']+\b", r)
    return _compare(sum(1 for w in words if w.isupper()), kw["capital_relation"], kw["capital_frequency"])


def _json_format(r, kw):
    v = r.strip()
    for prefix in ("```json", "```Json", "```JSON", "```"):
        if v.startswith(prefix):
            v = v[len(prefix):]
            break
    if v.endswith("```"):
        v = v[:-3]
    try:
        json.loads(v.strip())
    except ValueError:
        return False
    return True


def _multiple_sections(r, kw):
    sections = re.split(r"\s?" + re.escape(kw["section_spliter"]) + r"\s?\d+\s?", r)
    return len(sections) - 1 >= kw["num_sections"]


def _constrained_response(r, kw):
    v = r.strip()
    return any(o in v for o in ("My answer is yes.", "My answer is no.", "My answer is maybe."))


CHECKS = {
    "punctuation:no_comma": _no_comma,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
    "keywords:forbidden_words": _forbidden_words,
    "detectable_format:number_highlighted_sections": _highlighted_sections,
    "keywords:frequency": _frequency,
    "startend:quotation": _quotation,
    "change_case:english_lowercase": _lowercase,
    "keywords:existence": _existence,
    "detectable_format:title": _title,
    "keywords:letter_frequency": _letter_frequency,
    "detectable_format:number_bullet_lists": _bullet_lists,
    "detectable_content:number_placeholders": _placeholders,
    "length_constraints:number_paragraphs": _paragraphs,
    "startend:end_checker": _end_checker,
    "detectable_content:postscript": _postscript,
    "change_case:english_capital": _capital,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "detectable_format:json_format": _json_format,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:constrained_response": _constrained_response,
}
SUPPORTED = frozenset(CHECKS)


def check_instruction(instruction_id, kwargs, response):
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None}
    return bool(CHECKS[instruction_id](response, kw))


def check_prompt(row, response):
    each = [check_instruction(i, k, response) for i, k in zip(row["instruction_id_list"], row["kwargs"])]
    return all(each), each
