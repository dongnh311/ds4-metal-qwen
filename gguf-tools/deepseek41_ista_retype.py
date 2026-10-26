#!/usr/bin/env python3
"""Requantize the merged V4.1 GGUF into ds4-legal tensor types.

ds4's loader accepts only:
  * dense matmuls   : Q8_0 / Q4_K / Q4_0   (token_embd and *output_a/b must be
                      exactly F16 / Q8_0),
  * routed experts  : Q8_0 / IQ2_XXS / Q2_K / Q4_K / Q5_K / Q6_K / MXFP4,
  * misc            : exact F16 / F32 per tensor.
The ISTA RSC package instead ships Q5_K dense tiles, BF16 aux tensors and
IQ1_M/IQ2_XS routed experts.  This tool converts exactly the mismatched
tensors with the repo's reference dequantizers/quantizers
(`ds41_retype.c`: ported ggml dequant incl. iq1_m, quants.c for Q8_0 and
IQ2_XXS), appending them ahead of the Engram tables and shifting those
tables by a 16 KiB multiple, keeping every non-Engram rel stable.
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
CHUNK = 64 << 20
PLAN_MAGIC = b"DS4RTYP\x01"

F32, F16, BF16 = 0, 1, 2
Q4_0, Q8_0, Q2_K, Q4_K, Q5_K, Q6_K = 2, 8, 10, 12, 13, 14
IQ2_XXS, IQ2_XS, MXFP4, IQ1_M = 16, 17, 39, 29

# Fork-legal sets per role.
DENSE = {Q8_0, Q4_K, Q4_0}
ROUTED = {Q8_0, IQ2_XXS, Q2_K, Q4_K, Q5_K, Q6_K, MXFP4}
QWEN4_DENSE = {Q8_0, Q4_0, F16, BF16, F32}

GLOBALS = {
    "token_embd.weight": F16,
    "output.weight": Q8_0,
    "output_norm.weight": F32,
}

PER_LAYER = {
    "hc_attn_fn.weight": F16, "hc_ffn_fn.weight": F16,
    "hc_attn_base.weight": F32, "hc_attn_scale.weight": F32,
    "hc_ffn_base.weight": F32, "hc_ffn_scale.weight": F32,
    "attn_norm.weight": F32, "ffn_norm.weight": F32,
    "attn_sinks.weight": F32,
    "attn_q_a_norm.weight": F32, "attn_kv_a_norm.weight": F32,
    "attn_output_a.weight": Q8_0, "attn_output_b.weight": Q8_0,
    "attn_compressor_kv.weight": F16, "attn_compressor_gate.weight": F16,
    "attn_compressor_norm.weight": F32,
    "indexer.attn_k.weight": F16, "indexer.k_norm.weight": F32,
    "indexer.attn_q_b.weight": F16, "indexer.proj.weight": F16,
    "ffn_gate_inp.weight": F32, "exp_probs_b.bias": F32,
}
DENSE_ROLE = ("attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
              "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
              "ffn_down_shexp.weight")
ROUTED_ROLE = ("ffn_gate_exps.weight", "ffn_up_exps.weight",
               "ffn_down_exps.weight")


def die(msg):
    print(f"deepseek41_ista_retype: {msg}", file=sys.stderr)
    sys.exit(1)


def hu(n):
    return f"{n / 2 ** 30:.2f} GiB"


def align(x, a):
    return (x + a - 1) & ~(a - 1)


def lib():
    path = os.path.join(REPO, "gguf-tools", "libds41retype.dylib")
    if not os.path.exists(path):
        die(f"missing {path}; build: cc -O2 -shared -I. -Igguf-tools -o {path} "
            f"{REPO}/gguf-tools/ds41_retype.c {REPO}/gguf-tools/quants.c -lm")
    l = ctypes.CDLL(path)
    l.ds41_type_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    l.ds41_type_size.restype = ctypes.c_int64
    l.ds41_retype.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
                              ctypes.POINTER(ctypes.c_float)]
    l.ds41_retype.restype = ctypes.c_int
    return l


DOWN_Q2K = os.environ.get("DS41_DOWN_Q2K") == "1"
DENSE_Q4K = os.environ.get("DS41_DENSE_Q4K") == "1"


def target_type(name, ty):
    if name in GLOBALS:
        want = GLOBALS[name]
        if DENSE_Q4K and want == Q8_0:
            return Q4_K
        return want
    base = name
    if name.startswith("blk."):
        base = name.split(".", 2)[2]
    if base in PER_LAYER:
        return PER_LAYER[base]
    if base in DENSE_ROLE:
        if DENSE_Q4K and ty != Q4_K:
            return Q4_K
        return ty if ty in DENSE else Q8_0
    if base in ROUTED_ROLE:
        if DOWN_Q2K and base == "ffn_down_exps.weight" and ty != Q2_K:
            return Q2_K          # fork's streaming batch path needs Q2_K down
        return ty if ty in ROUTED else IQ2_XXS
    return ty       # norms / engram / unknown stay


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
    if "blk.1.engram_kv.weight" not in ents:
        die("engram tensors not fixed yet; run deepseek41_ista_engram_fix.py")
    tbl_abs = []
    for l in (1, 14):
        dims, ty, rel = ents[f"blk.{l}.engram_embd.weight"]
        tbl_abs.append(DATA_START + rel)
    old_tbl1 = min(tbl_abs)

    plan = []
    for name in names:
        dims, ty, rel = ents[name]
        want = target_type(name, ty)
        if want != ty:
            ne = 1
            for d in dims:
                ne *= d
            plan.append({"name": name, "dims": dims, "src_ty": ty,
                         "dst_ty": want, "src_rel": rel, "ne": ne,
                         "ncols": dims[0]})
    if not plan:
        die("nothing to convert")
    # reject unsupported conversions up front
    l = lib()
    for item in plan:
        sb = l.ds41_type_size(item["src_ty"], item["ne"])
        db = l.ds41_type_size(item["dst_ty"], item["ne"])
        if sb < 0 or db < 0:
            die(f"{item['name']}: unsupported {item['src_ty']} -> "
                f"{item['dst_ty']}")
        item["src_bytes"], item["dst_bytes"] = sb, db

    pos = old_tbl1
    for item in plan:
        pos = align(pos, 32)
        item["dst_pos"] = pos
        pos += item["dst_bytes"]
    d2 = align(pos - old_tbl1, 16384)
    new_tbl1 = old_tbl1 + d2

    table = bytearray()
    conv = {item["name"]: item for item in plan}
    for name in names:
        dims, ty, rel = ents[name]
        item = conv.get(name)
        if item is not None:
            nty, nrel = item["dst_ty"], item["dst_pos"] - DATA_START
        elif name.endswith("engram_embd.weight"):
            nty, nrel = ty, rel + d2
        else:
            nty, nrel = ty, rel
        nb = name.encode()
        table += struct.pack("<Q", len(nb)) + nb
        table += struct.pack("<I", len(dims))
        for d in dims:
            table += struct.pack("<Q", d)
        table += struct.pack("<I", nty)
        table += struct.pack("<Q", nrel)
    table = bytes(table)
    kv = os.pread(fd, kv_end - 24, 24)
    new_raw = 24 + len(kv) + len(table)
    if new_raw > DATA_START:
        die(f"header grows past data start ({new_raw} > {DATA_START})")
    hdr = bytearray()
    hdr += struct.pack("<I", m.GGUF_MAGIC) + struct.pack("<I", 3)
    hdr += struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv)
    hdr += kv + table
    hdr += b"\0" * (DATA_START - len(hdr))

    total_new = sum(item["dst_bytes"] for item in plan)
    print(f"plan: {len(plan)} tensors to convert, {hu(total_new)} of new data "
          f"(gap {hu(d2)}), tables -> +{d2}")
    kinds = {}
    for item in plan:
        kinds[(item["src_ty"], item["dst_ty"])] = \
            kinds.get((item["src_ty"], item["dst_ty"]), 0) + 1
    for (s, d), n in sorted(kinds.items()):
        print(f"  {s:3d} -> {d:3d}: {n} tensors")
    return {"n_tensors": n_tensors, "n_kv": n_kv, "data_start": DATA_START,
            "old_tbl1": old_tbl1, "new_tbl1": new_tbl1, "d2": d2,
            "eof_old": size, "hdr": bytes(hdr), "plan": plan}


def write_plan(path, p):
    blob = PLAN_MAGIC
    blob += struct.pack("<IQQQQQ", p["n_tensors"], p["n_kv"],
                        p["data_start"], p["old_tbl1"], p["new_tbl1"],
                        p["d2"])
    blob += struct.pack("<Q", p["eof_old"])
    blob += struct.pack("<Q", len(p["hdr"])) + p["hdr"]
    blob += struct.pack("<I", len(p["plan"]))
    for it in p["plan"]:
        nb = it["name"].encode()
        dims = it["dims"]
        blob += struct.pack("<Q", len(nb)) + nb
        blob += struct.pack("<I", len(dims))
        for d in dims:
            blob += struct.pack("<Q", d)
        blob += struct.pack("<iiQQQQQQ", it["src_ty"], it["dst_ty"],
                            it["src_rel"], it["ne"], it["ncols"],
                            it["src_bytes"], it["dst_bytes"], it["dst_pos"])
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_plan(path):
    blob = open(path, "rb").read()
    if not blob.startswith(PLAN_MAGIC):
        die("retype-plan is corrupt")
    off = len(PLAN_MAGIC)
    n_tensors, n_kv, data_start, old_tbl1, new_tbl1, d2 = \
        struct.unpack_from("<IQQQQQ", blob, off)
    off += struct.calcsize("<IQQQQQ")
    (eof_old,) = struct.unpack_from("<Q", blob, off)
    off += 8
    (hlen,) = struct.unpack_from("<Q", blob, off)
    off += 8
    hdr = blob[off:off + hlen]
    off += hlen
    (n,) = struct.unpack_from("<I", blob, off)
    off += 4
    plan = []
    for _ in range(n):
        (nlen,) = struct.unpack_from("<Q", blob, off)
        off += 8
        name = blob[off:off + nlen].decode()
        off += nlen
        (nd,) = struct.unpack_from("<I", blob, off)
        off += 4
        dims = struct.unpack_from(f"<{nd}Q", blob, off)
        off += 8 * nd
        src_ty, dst_ty, src_rel, ne, ncols, src_bytes, dst_bytes, dst_pos = \
            struct.unpack_from("<iiQQQQQQ", blob, off)
        off += struct.calcsize("<iiQQQQQQ")
        plan.append({"name": name, "dims": dims, "src_ty": src_ty,
                     "dst_ty": dst_ty, "src_rel": src_rel, "ne": ne,
                     "ncols": ncols, "src_bytes": src_bytes,
                     "dst_bytes": dst_bytes, "dst_pos": dst_pos})
    if off != len(blob):
        die("retype-plan trailing bytes")
    return {"n_tensors": n_tensors, "n_kv": n_kv, "data_start": data_start,
            "old_tbl1": old_tbl1, "new_tbl1": new_tbl1, "d2": d2,
            "eof_old": eof_old, "hdr": hdr, "plan": plan}


def journal_read(path):
    try:
        blob = open(path, "rb").read(9)
    except FileNotFoundError:
        return None
    if len(blob) != 9:
        die("retype-journal is truncated")
    return struct.unpack("<QB", blob)


def journal_write(path, low, state):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<QB", low, state))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def shift_tables(fd, out_path, p):
    side_path = out_path + ".retype-side"
    journal = out_path + ".retype-journal"
    d2 = p["d2"]
    st = journal_read(journal)
    if st is None:
        low, state = p["eof_old"], 0
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
    while low > p["old_tbl1"]:
        nb = min(CHUNK, low - p["old_tbl1"])
        move(low, nb)
        low -= nb
        done = p["eof_old"] - low
        if done % (2048 << 20) < CHUNK:
            print(f"  table shift {hu(done)} ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"  table shift complete in {time.time() - t0:.0f}s")
    journal_write(journal, p["old_tbl1"], 0)
    if os.path.exists(side_path):
        os.unlink(side_path)


def marker_dir(out_path):
    d = out_path + ".retype-done"
    os.makedirs(d, exist_ok=True)
    return d


def convert_item(fd, l, scratch, out_path, item):
    ne, ncols = item["ne"], item["ncols"]
    raw = os.pread(fd, item["src_bytes"], item["data_start"] if "data_start" in item else 0)
    return


def convert_indices(fd, p, out_path, idxs):
    l = lib()
    scratch = (ctypes.c_float * (1 << 20))()
    mdir = marker_dir(out_path)
    total = sum(p["plan"][i]["dst_bytes"] for i in idxs)
    done = 0
    t0 = time.time()
    for i in idxs:
        item = p["plan"][i]
        mk = os.path.join(mdir, f"{i:04d}")
        if os.path.exists(mk):
            done += item["dst_bytes"]
            continue
        print(f"  [{os.getpid()}] conv {item['name']}: {item['src_ty']}->"
              f"{item['dst_ty']} {hu(item['dst_bytes'])}", flush=True)
        ne, ncols = item["ne"], item["ncols"]
        raw = os.pread(fd, item["src_bytes"],
                       p["data_start"] + item["src_rel"])
        if len(raw) != item["src_bytes"]:
            die(f"{item['name']}: short read")
        src = ctypes.create_string_buffer(raw, item["src_bytes"])
        dst = ctypes.create_string_buffer(item["dst_bytes"])
        rows = ne // ncols
        per = max(1, (1 << 20) // ncols)
        r0 = 0
        while r0 < rows:
            nrows = min(per, rows - r0)
            elems = nrows * ncols
            s_off = l.ds41_type_size(item["src_ty"], r0 * ncols)
            d_off = l.ds41_type_size(item["dst_ty"], r0 * ncols)
            rc = l.ds41_retype(
                ctypes.cast(ctypes.addressof(src) + s_off, ctypes.c_char_p),
                item["src_ty"],
                ctypes.cast(ctypes.addressof(dst) + d_off, ctypes.c_char_p),
                item["dst_ty"], elems, ncols,
                ctypes.cast(scratch, ctypes.POINTER(ctypes.c_float)))
            if rc != 0:
                die(f"{item['name']}: retype failed ({rc}) at row {r0}")
            r0 += nrows
        os.pwrite(fd, dst.raw, item["dst_pos"])
        os.fsync(fd)
        with open(mk, "wb") as f:
            f.write(b"ok")
        done += item["dst_bytes"]
        el = time.time() - t0
        print(f"  [{os.getpid()}] {hu(done)}/{hu(total)} done in {el:.0f}s "
              f"({hu(done/max(el,1))}/s)", flush=True)


def header_is_new(fd, p):
    try:
        n_tensors, n_kv, kv_end, hdr_end, names, ents = parse(fd)
    except SystemExit:
        return False
    for item in p["plan"]:
        dims, ty, rel = ents.get(item["name"], (None, None, None))
        if ty != item["dst_ty"]:
            return False
        if rel != item["dst_pos"] - p["data_start"]:
            return False
    return True


def run(out_path, dry=False, worker=None, jobs=8):
    plan_path = out_path + ".retype-plan"
    journal_path = out_path + ".retype-journal"
    fresh = not os.path.exists(plan_path)
    if os.path.exists(plan_path):
        p = read_plan(plan_path)
        print("resume: reusing the existing retype-plan")
    else:
        fd = os.open(out_path, os.O_RDONLY)
        p = derive(fd)
        os.close(fd)
        if dry:
            print("dry-run: plan derived, no writes")
            return
        write_plan(plan_path, p)
    if worker is not None:
        idxs = [i for i in range(len(p["plan"])) if i % jobs == worker]
        fd = os.open(out_path, os.O_RDWR)
        try:
            convert_indices(fd, p, out_path, idxs)
        finally:
            os.close(fd)
        return
    fd = os.open(out_path, os.O_RDWR)
    try:
        if header_is_new(fd, p) and not os.path.exists(journal_path):
            print("already retyped; nothing to do")
        else:
            if os.path.exists(journal_path):
                shift_tables(fd, out_path, p)
            elif fresh:
                shift_tables(fd, out_path, p)
            elif os.fstat(fd).st_size != p["eof_old"] + p["d2"]:
                die("file size does not match the completed shift state")
            idxs = list(range(len(p["plan"])))
            convert_indices(fd, p, out_path, idxs)
            os.pwrite(fd, p["hdr"], 0)
            os.fsync(fd)
    finally:
        os.close(fd)
    for x in (journal_path, out_path + ".retype-side", plan_path):
        if os.path.exists(x):
            os.unlink(x)
    mdir = out_path + ".retype-done"
    if os.path.isdir(mdir):
        for f in os.listdir(mdir):
            os.unlink(os.path.join(mdir, f))
        os.rmdir(mdir)
    print(f"retyped: tables at {p['new_tbl1']}, gap {hu(p['d2'])}, file "
          f"{hu(p['eof_old'] + p['d2'])}")


def main(argv):
    out = None
    dry = False
    worker = None
    jobs = 8
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dry":
            dry = True
        elif a == "--worker":
            i += 1
            worker = int(argv[i])
            i += 1
            jobs = int(argv[i])
        elif a == "--jobs":
            i += 1
            jobs = int(argv[i])
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
    run(out, dry, worker, jobs)


if __name__ == "__main__":
    main(sys.argv[1:])