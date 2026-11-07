#!/usr/bin/env python3
"""Build a GGUF variant: tensor payloads picked per-name from two source GGUFs.

Usage: gguf_mix_variant.py BASE.gguf DONOR.gguf GROUPSPEC OUT.gguf
GROUPSPEC: comma list of fnmatch patterns; matching tensors come from DONOR,
all others from BASE. kv metadata is copied from BASE (types preserved).
"""
import sys, os, struct, fnmatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen4_pack_to_qwen4exp as P

# ggml block table: type -> (block_elems, block_bytes); (1, n) = n bytes/elem
BLOCK = {
    0:(1,4), 1:(1,2), 2:(32,18), 3:(32,20), 4:(1,2), 5:(1,4), 6:(32,22), 7:(32,24),
    8:(32,34), 9:(1,1), 10:(256,84), 11:(256,110), 12:(256,144), 13:(256,176),
    14:(256,210), 15:(256,292), 16:(256,66), 17:(256,74), 18:(256,98), 19:(256,50),
    20:(32,18), 21:(256,110), 22:(256,82), 23:(256,136), 24:(1,1), 25:(256,4),
    26:(256,4), 27:(1,8), 28:(256,6), 29:(256,56), 30:(1,2), 31:(1,2),
    32:(1,4), 33:(1,8), 34:(256,6), 35:(256,6), 36:(256,6), 37:(256,6),
    38:(256,6), 39:(32,17), 40:(256,6), 41:(256,6), 42:(64,18),
}
SCALAR_FMT = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<B",10:"<Q",11:"<q",12:"<d"}

def tbytes(tt, dims):
    n = 1
    for d in dims: n *= d
    be, bb = BLOCK[tt]
    assert n % be == 0, (tt, dims)
    return n // be * bb

def put_value(f, t, val):
    """Serialize one GGUF value of gguf-type t (Reader-parsed python value)."""
    if t == 9:
        assert isinstance(val, P.Arr)
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", val.item_type))
        f.write(struct.pack("<q", len(val.values)))
        for x in val.values:
            put_scalar(f, val.item_type, x)
        return
    f.write(struct.pack("<I", t))
    if t == 8:
        b = val.encode(); f.write(struct.pack("<q", len(b))); f.write(b)
    else:
        put_scalar(f, t, val)

def put_scalar(f, t, x):
    if t == 8:
        b = x.encode(); f.write(struct.pack("<q", len(b))); f.write(b)
    else:
        f.write(struct.pack(SCALAR_FMT[t], x))

def main():
    base_p, donor_p, spec, out_p = sys.argv[1:5]
    override = {}
    if len(sys.argv) > 5:
        import json
        override = json.load(open(sys.argv[5]))  # name -> {type, dims, file}
    pats = spec.split(",")
    rb, rd = P.Reader(base_p), P.Reader(donor_p)
    names = list(rb.tensors.keys())
    assert set(names) <= set(rd.tensors.keys()), set(names) - set(rd.tensors.keys())
    use_donor = {n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)}
    print(f"{len(use_donor)}/{len(names)} tensors from donor", flush=True)

    kv = rb.kv
    kvt = rb.kv_types
    f = open(out_p, "wb")
    f.write(b"GGUF"); f.write(struct.pack("<I", 3))
    f.write(struct.pack("<q", len(names)))
    f.write(struct.pack("<q", len(kv)))
    for k in kv:
        kb = k.encode(); f.write(struct.pack("<q", len(kb))); f.write(kb)
        put_value(f, kvt[k], kv[k])

    align = kv.get("general.alignment", 32)
    off = 0
    infos = []
    fb = open(base_p, "rb"); fd = open(donor_p, "rb")
    fover = {n: open(o["file"], "rb") for n, o in override.items()}
    for n in names:
        if n in override:
            o = override[n]
            tt, dims = o["type"], o["dims"]
            src, base, toff = fover[n], 0, 0
        elif n in use_donor:
            tt, dims, toff = rd.tensors[n]; src = fd; base = rd.data_start
        else:
            tt, dims, toff = rb.tensors[n]; src = fb; base = rb.data_start
        nb = tbytes(tt, dims)
        infos.append((n, dims, tt, off, nb, src, base + toff))
        off += (nb + align - 1) & ~(align - 1)
    for n, dims, tt, o, nb, src, pos in infos:
        f.write(struct.pack("<q", len(n.encode()))); f.write(n.encode())
        f.write(struct.pack("<I", len(dims)))
        for d in dims: f.write(struct.pack("<q", d))
        f.write(struct.pack("<I", tt)); f.write(struct.pack("<Q", o))
    data_start = (f.tell() + align - 1) & ~(align - 1)
    f.write(b"\0" * (data_start - f.tell()))
    CH = 256 * 1024 * 1024
    for n, dims, tt, o, nb, src, pos in infos:
        f.seek(data_start + o)
        src.seek(pos); rem = nb
        while rem:
            b = min(rem, CH); buf = src.read(b)
            assert len(buf) == b, (n, rem)
            f.write(buf); rem -= b
    f.close(); fb.close(); fd.close()
    print("wrote", out_p, os.path.getsize(out_p)/1e9, "GB", flush=True)

if __name__ == "__main__":
    main()
