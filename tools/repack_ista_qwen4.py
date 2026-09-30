"""Repack ISTA-DASLab's GSQ-RCO Qwen3.8-Flash-Next shard 1 for ds4, grafting the MTP layer (blk.N, N = ISTA's
block_count) and the ds4 metadata from Ivan's ds4 GGUF.

usage: python3 tools/repack_ista_qwen4.py ISTA_SHARD1.gguf IVAN.gguf OUT.gguf   (writes OUT and OUT.json)"""
import json
import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_lite as g  # noqa: E402

IVAN_KEYS = ["general.alignment", "qwen4exp.nextn_predict_layers", "qwen4exp.vocab_size",
             "qwen4exp.ple.row_count", "qwen4exp.ple.row_dimension", "qwen4exp.ple.seed",
             "qwen4exp.ple.vocab_base", "qwen4exp.ple.vocab_divisor"]
DROP = {"split.count", "split.no", "split.tensors.count"}
TO_F32 = re.compile(r"^blk\.\d+\.ffn_gate_inp(_shexp)?\.weight$")          # the loader requires F32
TO_F16 = re.compile(r"^(blk\.\d+\.hc_(attn|ffn)_(up|inject)|output_hc_up)\.weight$")   # the hc mixer takes F16/F32/Q8_0
BF16 = 30


def convert(raw, name):
    """BF16 raw bytes -> (type, bytes): F32 for the router, F16 for hc up/inject when every value round-trips."""
    f32 = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    if TO_F16.match(name):
        f16 = f32.astype(np.float16)
        if np.array_equal(f16.astype(np.float32), f32):
            return 1, f16.tobytes()
    return 0, f32.tobytes()


def plan(ista, ivan):
    n_layer = ista.kv["qwen4exp.block_count"][1]
    kv = {k: v for k, v in ista.kv.items() if k not in DROP}
    kv["qwen4exp.block_count"] = ivan.kv["qwen4exp.block_count"]
    ratios = ivan.kv["qwen4exp.attention.compress_ratios"]
    if ratios[1][1][:n_layer] != ista.kv["qwen4exp.attention.compress_ratios"][1][1]:
        raise ValueError("ISTA and Ivan disagree on the trunk's compress ratios")
    kv["qwen4exp.attention.compress_ratios"] = ratios
    for k in IVAN_KEYS:
        kv[k] = ivan.kv[k]
    kv["ds4.qwen4.down.logical_input"] = (g.T_U32, 640)
    kv["ds4.qwen4.down.physical_input"] = (g.T_U32, 640)
    tensors, conversions = [], {}
    for t in ista.tensors:
        out = {"name": t["name"], "dims": t["dims"], "type": t["type"], "nbytes": t["nbytes"],
               "src": (ista.path, t["abs"])}
        if t["type"] == BF16 and (TO_F32.match(t["name"]) or TO_F16.match(t["name"])):
            with open(ista.path, "rb") as f:
                f.seek(t["abs"])
                raw = f.read(t["nbytes"])
            ttype, data = convert(raw, t["name"])
            out.update(type=ttype, nbytes=len(data), data=data)
            del out["src"]
            conversions[t["name"]] = "BF16"
        tensors.append(out)
    mtp = re.compile(r"^blk\.(\d+)\.")
    for t in ivan.tensors:
        m = mtp.match(t["name"])
        if m and int(m.group(1)) >= n_layer:
            tensors.append({"name": t["name"], "dims": t["dims"], "type": t["type"], "nbytes": t["nbytes"],
                            "src": (ivan.path, t["abs"])})
    return kv, tensors, conversions


def main(argv):
    ista_path, ivan_path, out_path = argv[1], argv[2], argv[3]
    ista, ivan = g.Reader(ista_path), g.Reader(ivan_path)   # both refuse a truncated file
    kv, tensors, conversions = plan(ista, ivan)
    tmp = out_path + ".partial"
    try:
        shas = g.write(tmp, kv, tensors, kv["general.alignment"][1])
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, out_path)
    manifest = {
        "sources": {"ista": {"path": ista_path, "bytes": ista.size}, "ivan": {"path": ivan_path, "bytes": ivan.size}},
        "tensors": {t["name"]: {"type": t["type"], "bytes": t["nbytes"], "sha256": shas[t["name"]],
                                "converted_from": conversions.get(t["name"])} for t in tensors},
        "bytes": os.path.getsize(out_path),
    }
    pathlib.Path(out_path + ".json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote %s: %d tensors, %d converted, %.2f GiB" % (out_path, len(tensors), len(conversions),
                                                             manifest["bytes"] / 2 ** 30))


if __name__ == "__main__":
    main(sys.argv)
