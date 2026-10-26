#!/usr/bin/env python3
"""Give the merged V4.1 GGUF the engram tensors ds4's loader requires.

The ISTA RSC package ships, per Engram layer, `engram_q` / `engram_k` (BF16
(5120,4)) and `engram_wkv` (Q5_K (6144,25600)).  ds4 expects
`engram_q_norm` / `engram_k_norm` (F32 (5120,4)) and `engram_kv`
(F16 (6144,25600)) — same data, official dtype/layout.

ds4 also requires every resident tensor to end before the first Engram table
(engram extents are unmapped at shutdown).  The converted tensors therefore
need room *before* the tables: this tool shifts both engram tables forward by
a 16 KiB multiple and fills the freed gap with the six converted tensors.
Tensor rels outside the tables are untouched; the repo's own Q5_K reference
kernel (`ds41_q5k_dequant.c`) does the dequant with ggml semantics.

Crash safety mirrors deepseek41_ista_grow.py: a durable `.fix-plan` (new
tensor table + geometry) precedes any mutation, `.fix-journal`/`.fix-side`
make the 104 GiB table shift idempotent, and the gap rewrite reads only
untouched resident bytes so it can always be redone.
"""
import ctypes
import os
import struct
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "gguf-tools"))
import deepseek41_ista_merge as m   # noqa: E402

DATA_START = 5849088
LAYERS = (1, 14)
ROW_VALS = 6144 * 25600          # (24*256) x ((HC+1)*EMBD)
NORM_VALS = 5120 * 4             # EMBD x HC
KV_BYTES = ROW_VALS * 2          # F16
NORM_BYTES = NORM_VALS * 4       # F32
PLAN_MAGIC = b"DS4EFIX\x01"
CHUNK = 64 << 20

# name -> (new name, new ggml type)
RENAMES = {
    "engram_wkv": ("engram_kv", 1),        # F16
    "engram_q": ("engram_q_norm", 0),      # F32
    "engram_k": ("engram_k_norm", 0),      # F32
}
TABLE_SUFFIX = "engram_embd.weight"


def die(msg):
    print(f"deepseek41_ista_engram_fix: {msg}", file=sys.stderr)
    sys.exit(1)


def hu(n):
    return f"{n / 2 ** 30:.2f} GiB"


def align(x, a):
    return (x + a - 1) & ~(a - 1)


def lib():
    path = os.path.join(REPO, "gguf-tools", "libds41q5k.dylib")
    if not os.path.exists(path):
        die(f"missing {path}; build: cc -O2 -shared -o {path} "
            f"{REPO}/gguf-tools/ds41_q5k_dequant.c")
    l = ctypes.CDLL(path)
    l.ds41_q5k_to_f16.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                  ctypes.c_uint64]
    l.ds41_q5k_to_f16.restype = ctypes.c_int
    l.ds41_bf16_to_f32.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                   ctypes.c_uint64]
    return l


def parse(fd):
    c = m.Cursor(fd)
    if c.u32() != m.GGUF_MAGIC:
        die("not a GGUF file")
    if c.u32() != 3:
        die("expected GGUF v3")
    n_tensors, n_kv = c.u64(), c.u64()
    for _ in range(n_kv):
        c.string()
        m.skip_value(c, c.u32())
    kv_end = c.pos
    names, ents = [], {}
    for _ in range(n_tensors):
        k = c.string()
        name = k[1] if isinstance(k, tuple) else k
        nd = c.u32()
        dims = tuple(c.u64() for _ in range(nd))
        ty = c.u32()
        rel = c.u64()
        names.append(name)
        ents[name] = (dims, ty, rel)
    hdr_end = c.pos
    return n_tensors, n_kv, kv_end, hdr_end, names, ents


def derive(fd):
    size = os.fstat(fd).st_size
    n_tensors, n_kv, kv_end, hdr_end, names, ents = parse(fd)
    if n_tensors != 1010:
        die(f"expected 1010 tensors, found {n_tensors}")
    tbl_names = [f"blk.{l}.{TABLE_SUFFIX}" for l in LAYERS]
    tbl_abs = []
    for n in tbl_names:
        dims, ty, rel = ents[n]
        if ty != 24:
            die(f"{n}: unexpected type {ty}")
        tbl_abs.append(DATA_START + rel)
    old_tbl1 = min(tbl_abs)
    if tbl_abs[0] % 16384 or tbl_abs[1] % 16384:
        die("engram tables are not page aligned")
    src = {}
    for l in LAYERS:
        for part in ("engram_wkv", "engram_q", "engram_k"):
            n = f"blk.{l}.{part}.weight"
            dims, ty, rel = ents[n]
            if part == "engram_wkv":
                if dims != (6144, 25600) or ty != 13:
                    die(f"{n}: dims {dims} type {ty}")
                src[(l, part)] = (DATA_START + rel, ROW_VALS * 176 // 256)
            else:
                if dims != (5120, 4) or ty != 30:
                    die(f"{n}: dims {dims} type {ty}")
                src[(l, part)] = (DATA_START + rel, NORM_VALS * 2)
    # gap positions
    pos = old_tbl1
    where = {}
    for l in LAYERS:
        for part, nbytes in (("engram_kv", KV_BYTES),
                             ("engram_q_norm", NORM_BYTES),
                             ("engram_k_norm", NORM_BYTES)):
            pos = align(pos, 32)
            where[(l, part)] = pos
            pos += nbytes
    d2 = align(pos - old_tbl1, 16384)
    new_tbl1 = old_tbl1 + d2
    # new tensor table
    out = bytearray()
    for name in names:
        dims, ty, rel = ents[name]
        l = None
        for cand in LAYERS:
            if name.startswith(f"blk.{cand}."):
                l = cand
                break
        new_name, new_ty, new_rel = name, ty, rel
        if l is not None:
            base = name[len(f"blk.{l}."):]
            part = base[:-len(".weight")]
            if part in RENAMES:
                nn, nt = RENAMES[part]
                new_name = f"blk.{l}.{nn}.weight"
                new_ty = nt
                new_rel = where[(l, nn)] - DATA_START
            elif base == TABLE_SUFFIX:
                new_rel = rel + d2
        nb = new_name.encode()
        out += struct.pack("<Q", len(nb)) + nb
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", new_ty)
        out += struct.pack("<Q", new_rel)
    table = bytes(out)
    kv = os.pread(fd, kv_end - 24, 24)
    new_raw = 24 + len(kv) + len(table)
    if new_raw > DATA_START:
        die(f"header grows past data start ({new_raw} > {DATA_START})")
    hdr = bytearray()
    hdr += struct.pack("<I", m.GGUF_MAGIC)
    hdr += struct.pack("<I", 3)
    hdr += struct.pack("<Q", n_tensors)
    hdr += struct.pack("<Q", n_kv)
    hdr += kv + table
    hdr += b"\0" * (DATA_START - len(hdr))
    plan = {"n_tensors": n_tensors, "n_kv": n_kv, "data_start": DATA_START,
            "old_tbl1": old_tbl1, "new_tbl1": new_tbl1, "d2": d2,
            "eof_old": size, "table": table, "hdr": bytes(hdr),
            "where": where, "src": src}
    print(f"plan: gap {hu(d2)} ([{old_tbl1}, {new_tbl1})), tables -> "
          f"+{d2} B; {len(table)} B of tensor table; header {new_raw} B "
          f"(pad {DATA_START - new_raw})")
    return plan


def write_plan(path, plan):
    offsets = sorted(set(plan["where"].values()))
    blob = PLAN_MAGIC
    blob += struct.pack("<IQQQQQQ", plan["n_tensors"], plan["n_kv"],
                        plan["data_start"], plan["old_tbl1"],
                        plan["new_tbl1"], plan["d2"], plan["eof_old"])
    blob += struct.pack("<Q", len(plan["hdr"])) + plan["hdr"]
    blob += struct.pack("<I", len(plan["where"]))
    for (l, part), pos in plan["where"].items():
        blob += struct.pack("<H", l) + struct.pack("<Q", len(part)) \
            + part.encode() + struct.pack("<Q", pos)
    blob += struct.pack("<I", len(plan["src"]))
    for (l, part), (abs_off, nbytes) in plan["src"].items():
        blob += struct.pack("<H", l) + struct.pack("<Q", len(part)) \
            + part.encode() + struct.pack("<QQ", abs_off, nbytes)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_plan(path):
    blob = open(path, "rb").read()
    if not blob.startswith(PLAN_MAGIC):
        die("fix-plan is corrupt")
    off = len(PLAN_MAGIC)
    n_tensors, n_kv, data_start, old_tbl1, new_tbl1, d2, eof_old = \
        struct.unpack_from("<IQQQQQQ", blob, off)
    off += struct.calcsize("<IQQQQQQ")
    (hlen,) = struct.unpack_from("<Q", blob, off)
    off += 8
    hdr = blob[off:off + hlen]
    off += hlen
    (nw,) = struct.unpack_from("<I", blob, off)
    off += 4
    where = {}
    for _ in range(nw):
        (l,) = struct.unpack_from("<H", blob, off)
        off += 2
        (plen,) = struct.unpack_from("<Q", blob, off)
        off += 8
        part = blob[off:off + plen].decode()
        off += plen
        (pos,) = struct.unpack_from("<Q", blob, off)
        off += 8
        where[(l, part)] = pos
    (ns,) = struct.unpack_from("<I", blob, off)
    off += 4
    src = {}
    for _ in range(ns):
        (l,) = struct.unpack_from("<H", blob, off)
        off += 2
        (plen,) = struct.unpack_from("<Q", blob, off)
        off += 8
        part = blob[off:off + plen].decode()
        off += plen
        abs_off, nbytes = struct.unpack_from("<QQ", blob, off)
        off += 16
        src[(l, part)] = (abs_off, nbytes)
    if off != len(blob):
        die("fix-plan trailing bytes")
    return {"n_tensors": n_tensors, "n_kv": n_kv, "data_start": data_start,
            "old_tbl1": old_tbl1, "new_tbl1": new_tbl1, "d2": d2,
            "eof_old": eof_old, "hdr": hdr, "where": where, "src": src}


def journal_read(path):
    try:
        blob = open(path, "rb").read(9)
    except FileNotFoundError:
        return None
    if len(blob) != 9:
        die("fix-journal is truncated")
    low, state = struct.unpack("<QB", blob)
    return low, state


def journal_write(path, low, state):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<QB", low, state))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def shift_tables(fd, out_path, plan):
    side_path = out_path + ".fix-side"
    journal = out_path + ".fix-journal"
    low_end = plan["eof_old"]
    stop = plan["old_tbl1"]
    d2 = plan["d2"]
    st = journal_read(journal)
    if st is None:
        low = low_end
        state = 0
    else:
        low, state = st
        if state == 1 and not os.path.exists(side_path):
            die("journal says pending block but the side copy is missing")
        print(f"resume: shift frontier {low}, state {state}")

    def move(low, nb):
        nonlocal state
        P = low - nb
        if state == 0:
            blob = os.pread(fd, nb, P)
            if len(blob) != nb:
                die(f"short read at {P}")
            with open(side_path, "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            journal_write(journal, low, 1)
            state = 1
        else:
            blob = open(side_path, "rb").read()
            if len(blob) != nb:
                die("side copy length mismatch")
        os.pwrite(fd, blob, P + d2)
        os.fsync(fd)
        journal_write(journal, P, 0)
        state = 0
        if os.path.exists(side_path):
            os.unlink(side_path)

    t0 = time.time()
    base = low_end - low
    while low > stop:
        nb = min(CHUNK, low - stop)
        move(low, nb)
        low -= nb
        done = base + (low_end - low)
        if done % (2048 << 20) < CHUNK:
            print(f"  table shift {hu(done)} ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"  table shift complete in {time.time() - t0:.0f}s")
    journal_write(journal, stop, 0)
    if os.path.exists(side_path):
        os.unlink(side_path)


def write_gap(fd, plan):
    L = lib()
    d2 = plan["d2"]
    tail_from = max(pos for pos in plan["where"].values())
    tail_end = tail_from + max(KV_BYTES, NORM_BYTES)
    for l in LAYERS:
        src_off, nbytes = plan["src"][(l, "engram_wkv")]
        n = ROW_VALS
        src = os.pread(fd, nbytes, src_off)
        if len(src) != nbytes:
            die("short read of engram_wkv")
        dst = ctypes.create_string_buffer(KV_BYTES)
        rc = L.ds41_q5k_to_f16(src, dst, ctypes.c_uint64(n))
        if rc != 0:
            die(f"q5k dequant failed ({rc})")
        os.pwrite(fd, dst.raw, plan["where"][(l, "engram_kv")])
        os.fsync(fd)
        print(f"  layer {l}: engram_kv written "
              f"({hu(KV_BYTES)})", flush=True)
        for part, src_part in (("engram_q_norm", "engram_q"),
                               ("engram_k_norm", "engram_k")):
            so, nb = plan["src"][(l, src_part)]
            raw = os.pread(fd, nb, so)
            if len(raw) != nb:
                die(f"short read of {src_part} layer {l}")
            out = ctypes.create_string_buffer(NORM_BYTES)
            L.ds41_bf16_to_f32(raw, out, ctypes.c_uint64(NORM_VALS))
            os.pwrite(fd, out.raw, plan["where"][(l, part)])
        os.fsync(fd)
        print(f"  layer {l}: q_norm/k_norm written (F32)", flush=True)


def header_is_new(fd, plan):
    try:
        n_tensors, n_kv, kv_end, hdr_end, names, ents = parse(fd)
    except SystemExit:
        return False
    return "blk.1.engram_kv.weight" in ents


def fix(out_path, dry=False):
    plan_path = out_path + ".fix-plan"
    journal_path = out_path + ".fix-journal"
    if os.path.exists(plan_path):
        plan = read_plan(plan_path)
        print("resume: reusing the existing fix-plan")
    else:
        fd = os.open(out_path, os.O_RDONLY)
        plan = derive(fd)
        os.close(fd)
        if dry:
            print("dry-run: plan derived, no writes")
            return
        write_plan(plan_path, plan)
    d2 = plan["d2"]
    fd = os.open(out_path, os.O_RDWR)
    try:
        if header_is_new(fd, plan) and not os.path.exists(journal_path):
            print("already fixed; nothing to do")
        else:
            shift_tables(fd, out_path, plan)
            write_gap(fd, plan)
            os.pwrite(fd, plan["hdr"], 0)
            os.fsync(fd)
            want = plan["eof_old"] + d2
            got = os.fstat(fd).st_size
            if got < want:
                os.truncate(out_path, want)
    finally:
        os.close(fd)
    for p in (journal_path, out_path + ".fix-side", plan_path):
        if os.path.exists(p):
            os.unlink(p)
    print(f"fixed: tables at {plan['new_tbl1']}, gap {hu(d2)}, "
          f"file {hu(plan['eof_old'] + d2)}")


def main(argv):
    out = None
    dry = False
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dry":
            dry = True
        elif a == "--out":
            i += 1
            if i >= len(argv):
                die("--out needs a value")
            out = argv[i]
        elif a.startswith("-"):
            die(f"unknown option {a}")
        else:
            pos.append(a)
        i += 1
    base = pos[0] if pos else os.path.expanduser(
        "~/.local/share/ai-gateway/ds4-models/"
        "DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF")
    if not out:
        cands = [f for f in os.listdir(base) if f.endswith("-ds41.gguf")]
        if len(cands) != 1:
            die(f"cannot pick the merged file in {base}; pass --out")
        out = os.path.join(base, cands[0])
    fix(out, dry)


if __name__ == "__main__":
    main(sys.argv[1:])