#!/usr/bin/env python3
"""Grow a merged DeepSeek-V4.1 ISTA GGUF's metadata to ds4's required KV set.

ds4 validates the flat `deepseek41.*` config keys and the engram KV arrays
(token_map / primes / multipliers / pad_id / rows). The ISTA package carries
those objects only as tensors (`engram_token_map.weight` I32,
`engram_primes.weight` I64, `engram_multipliers.weight` I64), so the merged
file must carry them as metadata. The extra KV bytes (+~520 KB) exceed the
old 16 KiB page slack, so the aligned data start moves forward and every data
block shifts by a 16 KiB multiple. Tensor rels stay identical and the engram
tables keep their page alignment.

Crash safety:
  * `<out>.grow-plan`   exact header pieces (shard1 KVs, new KVs, tensor
                        table, target data start) written *before* touching
                        data — resume never needs to re-derive from files.
  * `<out>.grow-journal` 16 B: next source boundary (u64) + state (u8);
                        state 1 means `<out>.grow-side` holds the only
                        surviving copy of the in-flight 64 MiB block.
The shift is therefore idempotent: every resume either re-reads intact
source bytes or the durable side copy.
"""
import os
import struct
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "gguf-tools"))
import deepseek41_ista_merge as m   # noqa: E402

T_U32, T_F32, T_BOOL, T_STR, T_ARR, T_U64 = 4, 6, 7, 8, 9, 10
GGML_I32, GGML_I64 = 26, 27
DS4_N_VOCAB = 129280
DS4_N_LAYER = 40
CHUNK = 64 << 20
PLAN_MAGIC = b"DS4GROW\x01"

FLAT_INTS = [
    ("vocab_size", 129280), ("hidden_size", 5120),
    ("moe_intermediate_size", 2304), ("num_hidden_layers", 40),
    ("num_attention_heads", 64), ("num_key_value_heads", 1),
    ("head_dim", 512), ("qk_rope_head_dim", 64),
    ("q_lora_rank", 1280), ("o_lora_rank", 1024), ("o_groups", 8),
    ("n_routed_experts", 384), ("n_shared_experts", 1),
    ("num_experts_per_tok", 6), ("max_position_embeddings", 1048576),
    ("sliding_window", 128), ("index_n_heads", 32),
    ("index_head_dim", 128), ("index_topk", 512),
    ("candidate_source_layer_id", 20), ("candidate_topk_blocks", 2048),
    ("candidate_block_size", 8), ("hc_mult", 4),
    ("hc_sinkhorn_iters", 20), ("rope_theta", 10000),
    ("compress_rope_theta", 160000), ("engram.compressed_vocab_size", 99092),
]
FLAT_F32 = [
    ("rms_norm_eps", 1.0e-20), ("hc_eps", 1.0e-6),
    ("swiglu_limit", 10.0), ("routed_scaling_factor", 1.5),
    ("rope_scaling.factor", 16.0), ("rope_scaling.beta_fast", 32.0),
    ("rope_scaling.beta_slow", 1.0),
    ("rope_scaling.original_max_position_embeddings", 65536.0),
]
FLAT_STR = [
    ("scoring_func", "sqrtsoftplus"), ("hidden_act", "silu"),
    ("topk_method", "noaux_tc"),
]


def die(msg):
    print(f"deepseek41_ista_grow: {msg}", file=sys.stderr)
    sys.exit(1)


def hu(n):
    return f"{n / 2 ** 30:.2f} GiB"


def kv_frame(name, kind, payload):
    nb = name.encode()
    return struct.pack("<Q", len(nb)) + nb + struct.pack("<I", kind) + payload


def kv_u32(name, v):
    return kv_frame(name, T_U32, struct.pack("<I", v))


def kv_f32(name, v):
    return kv_frame(name, T_F32, struct.pack("<f", v))


def kv_str(name, v):
    b = v.encode()
    return kv_frame(name, T_STR, struct.pack("<Q", len(b)) + b)


def kv_bool(name, v):
    return kv_frame(name, T_BOOL, struct.pack("<B", int(bool(v))))


def kv_u32_arr(name, values):
    return kv_frame(name, T_ARR,
                    struct.pack("<IQ", T_U32, len(values))
                    + struct.pack(f"<{len(values)}I", *values))


def kv_u64_arr(name, values):
    return kv_frame(name, T_ARR,
                    struct.pack("<IQ", T_U64, len(values))
                    + struct.pack(f"<{len(values)}Q", *values))


def compress_ratios():
    # ds4_expected_layer_compress_ratio: il<2 -> 0, il<20 -> 2, else 1.
    return [0 if il < 2 else (2 if il < 20 else 1) for il in range(DS4_N_LAYER)]


def scan_header(fd):
    """Single KV walk: per-KV raw spans, pad token, engram rows, i32 arrays."""
    c = m.Cursor(fd)
    if c.u32() != m.GGUF_MAGIC:
        die("not a GGUF file")
    if c.u32() != 3:
        die("expected GGUF v3")
    n_tensors, n_kv = c.u64(), c.u64()
    spans = []
    pad_token_id = None
    rows = None
    i32_arrays = {}
    for i in range(n_kv):
        kv_start = c.pos
        k = c.string()
        name = k[1] if isinstance(k, tuple) else k
        t = c.u32()
        p0 = c.pos
        if name == "deepseek41.engram_pad_token_id":
            if t != T_U32:
                die("engram_pad_token_id is not u32")
            pad_token_id = struct.unpack("<I", os.pread(fd, 4, p0))[0]
            c.pos = p0 + 4
        elif name == "deepseek41.engram.num_embeddings":
            if t != T_ARR:
                die("engram.num_embeddings is not an array")
            at = c.u32()
            cnt = c.u64()
            if cnt != 2:
                die("engram.num_embeddings must carry 2 rows")
            fmt = {4: "I", 5: "i", 10: "Q"}.get(at)
            if fmt is None:
                die(f"engram.num_embeddings element type {at}")
            rows = struct.unpack(f"<2{fmt}", os.pread(fd, 2 * struct.calcsize(fmt), c.pos))
            c.pos += 2 * struct.calcsize(fmt)
            if rows[0] < 0 or rows[1] < 0:
                die("negative engram row count")
        else:
            if t == T_ARR:
                at = struct.unpack("<I", os.pread(fd, 4, p0))[0]
                if at in (4, 5) and name in (
                        "deepseek41.kv_source_layer_ids",
                        "deepseek41.index_source_layer_ids",
                        "deepseek41.engram.layer_ids"):
                    cnt = struct.unpack("<Q", os.pread(fd, 8, p0 + 4))[0]
                    vals = struct.unpack(
                        f"<{cnt}i", os.pread(fd, cnt * 4, p0 + 12))
                    i32_arrays[name] = list(vals)
            m.skip_value(c, t)
        spans.append((name, kv_start, c.pos))
    kv_end = c.pos
    return n_kv, n_tensors, kv_end, spans, pad_token_id, rows, i32_arrays


def tensor_table(fd, kv_end, n_tensors):
    c = m.Cursor(fd, kv_end)
    ents = {}
    for _ in range(n_tensors):
        k = c.string()
        name = k[1] if isinstance(k, tuple) else k
        nd = c.u32()
        dims = [c.u64() for _ in range(nd)]
        ty = c.u32()
        rel = c.u64()
        ents[name] = (dims, ty, rel)
    return ents, c.pos


def tensor_bytes(fd, data_start, ents, name, elem_bytes, expect_count=None):
    dims, ty, rel = ents.get(name, (None, None, None))
    if dims is None:
        die(f"merged file lacks {name}")
    if ty not in (GGML_I32, GGML_I64):
        die(f"{name}: unexpected ggml type {ty}")
    if (ty == GGML_I32) != (elem_bytes == 4):
        die(f"{name}: element size {elem_bytes} disagrees with type {ty}")
    total = 1
    for d in dims:
        total *= d
    if expect_count is not None and total != expect_count:
        die(f"{name}: {dims} carries {total} values, expected "
            f"{expect_count}")
    raw = os.pread(fd, total * elem_bytes, data_start + rel)
    if len(raw) != total * elem_bytes:
        die(f"{name}: short read at 0x{data_start + rel:x}")
    return raw


def derive(fd, size, data_start):
    n_kv, n_tensors, kv_end, spans, pad_token_id, rows, i32_arrays = \
        scan_header(fd)
    if n_kv != 83:
        die(f"expected the current merged header (83 KVs), found {n_kv}")
    ents, hdr_end = tensor_table(fd, kv_end, n_tensors)
    if min(e[2] for e in ents.values()) != 0:
        die("first tensor rel is not 0")
    slack = data_start - hdr_end
    if slack <= 0:
        die("header already exceeds the data start")

    token_map_raw = tensor_bytes(fd, data_start, ents,
                                 "engram_token_map.weight", 4, DS4_N_VOCAB)
    token_map = struct.unpack(f"<{DS4_N_VOCAB}I", token_map_raw)
    primes_raw = tensor_bytes(fd, data_start, ents, "engram_primes.weight",
                              8, 48)
    mult_raw = tensor_bytes(fd, data_start, ents,
                            "engram_multipliers.weight", 8, 8)

    nsep = len(set(token_map))
    if nsep != 99092:
        die(f"token_map compresses to {nsep} ids, ds4 expects 99092")
    if max(token_map) != nsep - 1:
        die("token_map is not dense 0..N-1")
    if pad_token_id is None or pad_token_id >= DS4_N_VOCAB:
        die(f"bad engram_pad_token_id {pad_token_id}")
    pad_id = token_map[pad_token_id]
    primes = struct.unpack("<48Q", primes_raw)
    if [sum(primes[:24]), sum(primes[24:])] != list(rows):
        die(f"prime bucket sums {sum(primes[:24])}/{sum(primes[24:])} != "
            f"engram rows {rows}")
    multipliers = struct.unpack("<8Q", mult_raw)
    if any(v % 2 == 0 for v in multipliers):
        die("engram multipliers must all be odd")

    # ds4 requires u32 arrays; the package stores these three as i32.
    DROP = {"deepseek41.kv_source_layer_ids",
            "deepseek41.index_source_layer_ids",
            "deepseek41.engram.layer_ids"}
    want = {
        "deepseek41.kv_source_layer_ids": [2, 8, 14, 20],
        "deepseek41.index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "deepseek41.engram.layer_ids": [1, 14],
    }
    for name, expect in want.items():
        got = i32_arrays.get(name)
        if got is None:
            die(f"merged file lacks the i32 array {name}")
        if got != expect:
            die(f"{name} = {got} != ds4 expectation {expect}")

    shard1_kv = b"".join(os.pread(fd, e - s, s)
                         for (n, s, e) in spans if n not in DROP)
    extras = [
        kv_str("deepseek41.engram.encoding", m.ENCODING_NAME),
        kv_u32("general.alignment", m.GGUF_ALIGNMENT),
    ]
    for name in ("deepseek41.kv_source_layer_ids",
                 "deepseek41.index_source_layer_ids",
                 "deepseek41.engram.layer_ids"):
        extras.append(kv_u32_arr(name, want[name]))
    for k, v in FLAT_INTS:
        extras.append(kv_u32(f"deepseek41.{k}", v))
    for k, v in FLAT_F32:
        extras.append(kv_f32(f"deepseek41.{k}", v))
    for k, v in FLAT_STR:
        extras.append(kv_str(f"deepseek41.{k}", v))
    extras.append(kv_bool("deepseek41.norm_topk_prob", True))
    extras.append(kv_u32_arr("deepseek41.engram.rows", list(rows)))
    extras.append(kv_u32_arr("deepseek41.compress_ratios", compress_ratios()))
    extras.append(kv_u32("deepseek41.engram.pad_id", pad_id))
    extras.append(kv_u32_arr("deepseek41.engram.token_map", list(token_map)))
    extras.append(kv_u32_arr("deepseek41.engram.primes", list(primes)))
    extras.append(kv_u64_arr("deepseek41.engram.multipliers",
                             list(multipliers)))
    extras_blob = b"".join(extras)
    table = os.pread(fd, hdr_end - kv_end, kv_end)
    new_raw = 24 + len(shard1_kv) + len(extras_blob) + len(table)
    a = m.GGUF_ALIGNMENT
    new_data_start = (new_raw + a - 1) & ~(a - 1)
    plan = {
        "n_tensors": n_tensors, "n_kv": len(spans) - len(DROP) + len(extras),
        "new_data_start": new_data_start, "data_start": data_start,
        "shard1_kv": shard1_kv, "extras": extras_blob, "table": table,
        "size": size,
    }
    print(f"derive: n_kv 83 -> {plan['n_kv']} (+{len(extras)} -{len(DROP)}), "
          f"extras {len(extras_blob)} B, token_map {len(token_map_raw)} B, "
          f"data_start {data_start} -> {new_data_start} "
          f"(delta {new_data_start - data_start})")
    return plan


def write_plan(path, plan):
    a = m.GGUF_ALIGNMENT
    blob = PLAN_MAGIC
    blob += struct.pack("<IQQQQQ", plan["n_tensors"], plan["n_kv"],
                        plan["data_start"], plan["new_data_start"],
                        plan["size"], a)
    for key in ("shard1_kv", "extras", "table"):
        blob += struct.pack("<Q", len(plan[key])) + plan[key]
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_plan(path):
    with open(path, "rb") as f:
        blob = f.read()
    if not blob.startswith(PLAN_MAGIC):
        die("grow-plan is corrupt")
    off = len(PLAN_MAGIC)
    n_tensors, n_kv, data_start, new_data_start, size, _a = \
        struct.unpack_from("<IQQQQQ", blob, off)
    off += struct.calcsize("<IQQQQQ")
    plan = {"n_tensors": n_tensors, "n_kv": n_kv,
            "data_start": data_start, "new_data_start": new_data_start,
            "size": size}
    for key in ("shard1_kv", "extras", "table"):
        (n,) = struct.unpack_from("<Q", blob, off)
        off += 8
        plan[key] = blob[off:off + n]
        off += n
    if off != len(blob):
        die("grow-plan trailing bytes")
    return plan


def build_header(plan):
    out = bytearray()
    out += struct.pack("<I", m.GGUF_MAGIC)
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", plan["n_tensors"])
    out += struct.pack("<Q", plan["n_kv"])
    out += plan["shard1_kv"] + plan["extras"] + plan["table"]
    new_data_start = plan["new_data_start"]
    if len(out) > new_data_start:
        die("header exceeds the planned data start")
    out += b"\0" * (new_data_start - len(out))
    return bytes(out)


def journal_read(path):
    try:
        with open(path, "rb") as f:
            blob = f.read(9)
    except FileNotFoundError:
        return None
    if len(blob) != 9:
        die("grow-journal is truncated")
    low, state = struct.unpack("<QB", blob)
    return low, state


def journal_write(path, low, state):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<QB", low, state))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def shift(fd, out_path, plan):
    data_start = plan["data_start"]
    size = plan["size"]
    delta = plan["new_data_start"] - data_start
    side_path = out_path + ".grow-side"
    journal = out_path + ".grow-journal"
    state = journal_read(journal)
    if state is None:
        low = size
        state = 0
    else:
        low, state = state
        if state == 1 and not os.path.exists(side_path):
            die("journal says pending block but the side copy is missing")
        print(f"resume: low={hu(low - data_start)} of "
              f"{hu(size - data_start)} left, state={state}")

    def doside(low, nb):
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
            with open(side_path, "rb") as f:
                blob = f.read()
            if len(blob) != nb:
                die("side copy length mismatch")
        os.pwrite(fd, blob, P + delta)
        os.fsync(fd)
        journal_write(journal, P, 0)
        state = 0
        if os.path.exists(side_path):
            os.unlink(side_path)

    t0 = time.time()
    moved0 = size - low
    while low > data_start:
        nb = min(CHUNK, low - data_start)
        doside(low, nb)
        low -= nb
        moved = size - low
        if (moved - moved0) % (1024 << 20) < CHUNK:
            el = time.time() - t0
            print(f"  shift {hu(moved - moved0)} left "
                  f"{hu(low - data_start)} ({el:.0f}s)", flush=True)
    print(f"  shift complete in {time.time() - t0:.0f}s")
    # Leave the journal at (data_start, 0) as proof; the caller deletes it
    # only after the new header is fsync'd.
    journal_write(journal, data_start, 0)
    if os.path.exists(side_path):
        os.unlink(side_path)


def header_n_kv(fd):
    c = m.Cursor(fd)
    if c.u32() != m.GGUF_MAGIC:
        die("not a GGUF file")
    c.u32()
    c.u64()
    return c.u64()


def grow(out_path, dry=False):
    plan_path = out_path + ".grow-plan"
    journal_path = out_path + ".grow-journal"
    if os.path.exists(plan_path):
        plan = read_plan(plan_path)
        print("resume: reusing the existing grow-plan")
    else:
        fd = os.open(out_path, os.O_RDONLY)
        size = os.fstat(fd).st_size
        plan = derive(fd, size, 5341184)
        os.close(fd)
        if dry:
            print("dry-run: plan derived, no writes")
            return
        write_plan(plan_path, plan)
    delta = plan["new_data_start"] - plan["data_start"]
    fd = os.open(out_path, os.O_RDWR)
    try:
        current_kv = header_n_kv(fd)
        if current_kv == plan["n_kv"] and not os.path.exists(journal_path):
            print("header already grown; nothing to do")
        else:
            if current_kv != 83 and not os.path.exists(journal_path):
                die(f"header has {current_kv} KVs; expected 83 before "
                    "the shift")
            shift(fd, out_path, plan)
            hdr = build_header(plan)
            os.pwrite(fd, hdr, 0)
            os.fsync(fd)
    finally:
        os.close(fd)
    if os.path.exists(journal_path):
        os.unlink(journal_path)
    if os.path.exists(out_path + ".grow-side"):
        os.unlink(out_path + ".grow-side")
    os.unlink(plan_path)
    print(f"grown: n_kv={plan['n_kv']}, data_start={plan['new_data_start']}, "
          f"file={hu(plan['size'] + delta)}")


def main(argv):
    src = None
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
    grow(out, dry)


if __name__ == "__main__":
    main(sys.argv[1:])