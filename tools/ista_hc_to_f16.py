"""Rewrite a repacked ISTA Qwen3.8 GGUF with its F32 hc mixer weights (hc_{attn,ffn}_{up,inject}, output_hc_up)
in F16, by repack_ista_qwen4's rule. Every other tensor and all metadata are copied unchanged.

usage: python3 tools/ista_hc_to_f16.py IN.gguf OUT.gguf   (writes OUT and OUT.json; reads IN.json if present)"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_lite as g  # noqa: E402
import repack_ista_qwen4 as rp  # noqa: E402

F32, F16 = 0, 1


def plan(src):
    """-> (tensors for gguf_lite.write, names converted to F16)"""
    tensors, converted = [], set()
    for t in src.tensors:
        out = {"name": t["name"], "dims": t["dims"], "type": t["type"], "nbytes": t["nbytes"],
               "src": (src.path, t["abs"])}
        if t["type"] == F32 and rp.TO_F16.match(t["name"]):
            with open(src.path, "rb") as f:
                f.seek(t["abs"])
                data = rp.hc_f16(np.frombuffer(f.read(t["nbytes"]), dtype=np.float32))
            if data is not None:
                out.update(type=F16, nbytes=len(data), data=data)
                del out["src"]
                converted.add(t["name"])
        tensors.append(out)
    return tensors, converted


def main(argv):
    in_path, out_path = argv[1], argv[2]
    src = g.Reader(in_path)   # refuses a truncated file
    prior = {}
    if os.path.exists(in_path + ".json"):
        prior = json.loads(pathlib.Path(in_path + ".json").read_text()).get("tensors", {})
    tensors, converted = plan(src)
    tmp = out_path + ".partial"
    try:
        shas = g.write(tmp, src.kv, tensors, src.alignment)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, out_path)

    def origin(name):
        was = prior.get(name, {}).get("converted_from")
        return was if was or name not in converted else "F32"

    manifest = {
        "sources": {"input": {"path": in_path, "bytes": src.size}},
        "tensors": {t["name"]: {"type": t["type"], "bytes": t["nbytes"], "sha256": shas[t["name"]],
                                "converted_from": origin(t["name"])} for t in tensors},
        "bytes": os.path.getsize(out_path),
    }
    pathlib.Path(out_path + ".json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote %s: %d tensors, %d hc tensors to F16, %.2f GiB" % (out_path, len(tensors), len(converted),
                                                                    manifest["bytes"] / 2 ** 30))


if __name__ == "__main__":
    main(sys.argv)
