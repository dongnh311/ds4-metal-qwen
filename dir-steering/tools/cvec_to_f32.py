#!/usr/bin/env python3
"""Convert a llama.cpp control-vector GGUF into a ds4 directional-steering file.

python3 dir-steering/tools/cvec_to_f32.py --in refusal.gguf --layers 4-44 --out refusal-4-44.f32

llama.cpp applies tensor direction.N at layer index N (0-based; there is never a direction.0). ds4 reads
one f32 row per trunk layer. Row N of the output is direction.N scaled to unit length for N in --layers,
and zero elsewhere: ds4 projects x -= s * (x.d) * d, which assumes unit rows and leaves the layer of a
zero row untouched. A JSON sidecar records the source, the range and each source row's norm.
"""
import argparse
import hashlib
import json
import math
import pathlib
import struct

GGUF_F32 = 0
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
           12: "<d"}
_STRING, _ARRAY = 8, 9


class _Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, fmt):
        (value,) = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += struct.calcsize(fmt)
        return value

    def string(self):
        n = self.take("<Q")
        value = self.data[self.pos:self.pos + n].decode("utf-8")
        self.pos += n
        return value

    def value(self, kind):
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            item = self.take("<I")
            n = self.take("<Q")
            return [self.value(item) for _ in range(n)]
        if kind not in _SCALAR:
            raise ValueError("unknown GGUF value type %d" % kind)
        return self.take(_SCALAR[kind])


def read_gguf(data):
    """(key/values, {tensor name: (dims, type, absolute byte offset)}) of a GGUF v2/v3 file."""
    if data[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    r = _Reader(data)
    r.pos = 4
    version = r.take("<I")
    if version not in (2, 3):
        raise ValueError("unsupported GGUF version %d" % version)
    n_tensors = r.take("<Q")
    n_kv = r.take("<Q")
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        kv[key] = r.value(r.take("<I"))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.take("<Q") for _ in range(r.take("<I"))]
        kind = r.take("<I")
        offset = r.take("<Q")
        infos.append((name, dims, kind, offset))
    align = int(kv.get("general.alignment", 32))
    base = (r.pos + align - 1) // align * align
    return kv, {name: (dims, kind, base + offset) for name, dims, kind, offset in infos}


def parse_layers(text):
    lo, _, hi = text.partition("-")
    lo, hi = int(lo), int(hi or lo)
    if lo < 0 or hi < lo:
        raise ValueError("bad layer range %r" % text)
    return lo, hi


def convert(data, layers, n_layers, width):
    """The ds4 rows (n_layers lists of width floats) and the norm of each converted source row."""
    kv, tensors = read_gguf(data)
    arch = kv.get("general.architecture")
    if arch != "controlvector":
        raise ValueError("general.architecture is %r, not 'controlvector'" % arch)
    lo, hi = layers
    if hi >= n_layers:
        raise ValueError("layer range %d-%d is outside the model's %d layers" % (lo, hi, n_layers))
    rows = [[0.0] * width for _ in range(n_layers)]
    norms = {}
    for layer in range(lo, hi + 1):
        name = "direction.%d" % layer
        if name not in tensors:
            raise ValueError("%s is missing" % name)
        dims, kind, offset = tensors[name]
        if kind != GGUF_F32 or dims != [width]:
            raise ValueError("%s is type %d with dims %s, expected f32 [%d]" % (name, kind, dims, width))
        values = struct.unpack_from("<%df" % width, data, offset)
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("%s has norm %r" % (name, norm))
        rows[layer] = [v / norm for v in values]
        norms[layer] = norm
    return rows, norms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="control-vector GGUF")
    ap.add_argument("--out", required=True, help="ds4 steering file to write (.f32)")
    ap.add_argument("--layers", required=True, help="inclusive 0-based layer range, e.g. 4-44")
    ap.add_argument("--n-layers", type=int, default=48, help="trunk layers (Qwen3.8-Flash-Next: 48)")
    ap.add_argument("--width", type=int, default=2560, help="hidden width (Qwen3.8-Flash-Next: 2560)")
    args = ap.parse_args(argv)
    data = pathlib.Path(args.src).read_bytes()
    try:
        lo, hi = parse_layers(args.layers)
        rows, norms = convert(data, (lo, hi), args.n_layers, args.width)
    except ValueError as e:
        raise SystemExit("cvec_to_f32: %s" % e)
    blob = b"".join(struct.pack("<%df" % args.width, *row) for row in rows)
    out = pathlib.Path(args.out)
    out.write_bytes(blob)
    meta = {"source": str(pathlib.Path(args.src).resolve()), "source_sha256": hashlib.sha256(data).hexdigest(),
            "output_sha256": hashlib.sha256(blob).hexdigest(), "layers": [lo, hi],
            "n_layers": args.n_layers, "width": args.width,
            "source_norms": {str(k): round(v, 6) for k, v in sorted(norms.items())}}
    pathlib.Path(str(out) + ".json").write_text(json.dumps(meta, indent=1) + "\n")
    print("%s: %d rows, layers %d-%d steered, source norms %.4f..%.4f" % (
        out, args.n_layers, lo, hi, min(norms.values()), max(norms.values())))


if __name__ == "__main__":
    main()
