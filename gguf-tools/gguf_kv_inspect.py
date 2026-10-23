#!/usr/bin/env python3
"""Dump a GGUF's key-value metadata and tensor table (CPU-only, no model load).

Usage: python3 gguf_kv_inspect.py FILE [--skip tokenizer.ggml.* gguf.facebook.llm]
Tokenizer vocab/score/merges/hash arrays are counted, not printed, unless the
key matches --show-pat.
"""
import sys
from gguf import GGUFReader, GGUFValueType  # GGMLQuantizationType: export name varies by gguf-py version
try:
    from gguf import GGMLQuantizationType as _QType
except ImportError:
    from gguf.core import GGMLQuantizationType as _QType  # type: ignore

argv = sys.argv
if len(argv) < 2:
    raise SystemExit("usage: gguf_kv_inspect.py FILE [--show PAT ...]")
path = argv[1]
args = argv[2:]
show_pats = []
if args and args[0] == "--show":
    show_pats = args[1:]

SKIP_PREFIXES = ("gguf.", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
                 "tokenizer.ggml.merges", "tokenizer.ggml.scores", "tokenizer.ggml.hash",
                 "tokenizer.huggingface.", "template.")


def fmt_scalar(v):
    if isinstance(v, bytes):
        return repr(v.decode("utf-8", "replace"))
    if isinstance(v, float):
        return str(v)
    return str(v)


def print_kv(k, f):
    t = f.types[0]
    if t == GGUFValueType.STRING:
        vals = f.parts[-1]
        s = bytes(vals[0]) if len(vals) else b""
        print(f"{k}: str = {s.decode('utf-8', 'replace')[:200]!r}")
    elif t == GGUFValueType.ARRAY:
        et = f.types[1]
        n = len(f.parts[-1])
        if et == GGUFValueType.STRING:
            print(f"{k}: str[{n}]" + (f" head={[bytes(v).decode('utf-8','replace') for v in f.parts[-1][:4]]}" if n <= 8 else ""))
        else:
            head = [fmt_scalar(v) for v in f.parts[-1][:8]]
            print(f"{k}: {et.name}[{n}] head={head}")
    else:
        head = [fmt_scalar(v) for v in f.parts[-1][:8]]
        print(f"{k}: {t.name} = {head}" if len(head) > 1 else f"{k}: {t.name} = {head[0]}")


r = GGUFReader(path)
print(f"file={path} n_tensors={len(r.tensors)} data_offset={r.data_offset}")
for k, f in r.fields.items():
    if k.startswith("gguf."):
        continue
    if any(k.startswith(p) for p in SKIP_PREFIXES) and not any(k.startswith(p) for p in show_pats):
        t = f.types[0]
        n = len(f.parts[-1]) if t in (GGUFValueType.ARRAY, GGUFValueType.STRING) else 1
        print(f"{k}: [{t.name} x {n}] skipped")
    else:
        print_kv(k, f)
print()
print("== tensors ==")
for t in r.tensors:
    qt = _QType(t.tensor_type).name
    dims = "x".join(str(d) for d in reversed(t.shape))
    print(f"  {t.name:44s} {qt:10s} {dims}")
for key in show_pats:
    f = r.fields.get(key)
    if f is not None and f.types[0] == GGUFValueType.ARRAY and f.types[1] == GGUFValueType.STRING:
        base = f.parts[-1]
        if len(base) > 248077:
            print(f"token 248077 = {bytes(base[248077])!r}")
