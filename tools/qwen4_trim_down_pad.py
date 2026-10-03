"""Rewrite a qwen4exp GGUF whose routed down experts are Q4_K rows padded to whole 256-value super-blocks
(dims [roundup256(F), n_embd, n_expert]) as trimmed rows of F values: each row keeps its first
gguf_lite.q4k_row_bytes(F) bytes (whole blocks, then the last block's 16-byte header and the qs chunks of the
real values). The values the kernels read are unchanged; every other tensor and all metadata are copied as is.

usage: python3 tools/qwen4_trim_down_pad.py IN.gguf OUT.gguf   (writes OUT and OUT.json)"""
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_lite as g  # noqa: E402

Q4_K = 12
DOWN = re.compile(r"blk\.\d+\.ffn_down_exps\.weight$")
trim_rows = g.trim_rows


def plan(src):
    """-> (tensors for gguf_lite.write, {name: original dims} of the trimmed tensors)"""
    arch = src.kv.get("general.architecture", (None, None))[1]
    if arch != "qwen4exp":
        raise SystemExit("%s: architecture %r, expected qwen4exp" % (src.path, arch))
    ff = src.kv.get("qwen4exp.expert_feed_forward_length", (None, 0))[1]
    if not ff or ff % 64 or ff % 256 == 0:
        raise SystemExit("%s: expert_feed_forward_length %s is not a multiple of 64 that ends inside a "
                         "256-value block" % (src.path, ff))
    padded = (ff + 255) // 256 * 256
    in_row, out_row = g.q4k_row_bytes(padded), g.q4k_row_bytes(ff)
    tensors, trimmed = [], {}
    for t in src.tensors:
        out = {"name": t["name"], "dims": t["dims"], "type": t["type"], "nbytes": t["nbytes"],
               "src": (src.path, t["abs"])}
        if t["type"] == Q4_K and DOWN.match(t["name"]) and t["dims"][0] == padded:
            rows = t["nbytes"] // in_row
            out.update(dims=[ff] + t["dims"][1:], nbytes=rows * out_row, trim=(src.path, t["abs"], in_row, out_row))
            del out["src"]
            trimmed[t["name"]] = t["dims"]
        tensors.append(out)
    return tensors, trimmed


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    in_path, out_path = argv[1], argv[2]
    src = g.Reader(in_path)   # refuses a truncated file
    tensors, trimmed = plan(src)
    if not trimmed:
        sys.exit("%s: no padded Q4_K routed down tensor; nothing to trim, nothing written" % in_path)
    tmp = out_path + ".partial"
    try:
        shas = g.write(tmp, src.kv, tensors, src.alignment, nocache=True)   # 100 GB of I/O off the page cache
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, out_path)
    manifest = {
        "sources": {"input": {"path": in_path, "bytes": src.size}},
        "tensors": {t["name"]: {"type": t["type"], "dims": t["dims"], "bytes": t["nbytes"],
                                "sha256": shas[t["name"]], "trimmed_from": trimmed.get(t["name"])}
                    for t in tensors},
        "bytes": os.path.getsize(out_path),
    }
    pathlib.Path(out_path + ".json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote %s: %d tensors, %d down tensors trimmed, %.2f GiB (input %.2f GiB)" % (
        out_path, len(tensors), len(trimmed), manifest["bytes"] / 2 ** 30, src.size / 2 ** 30))


if __name__ == "__main__":
    main(sys.argv)
