#!/usr/bin/env python3
"""
deepseek41_ista_merge.py — merge the ISTA-DASLab llama.cpp-split
DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XXS shards plus its IQ4_NL Engram
companion into ONE single-file ds4 GGUF.

The wrapping roles (GGUF header building, on-disk layout, resume,
deletes) live here; row conversion calls the fork's own encoder via
libds41ista.dylib (built from gguf-tools/deepseek41_ista_core.c, which
links ds4_engram.c's ds4_engram_encode_row136()).  A startup self-test
cross-checks the C kernel against an independent pure-Python float32
reference (RNE everywhere, math identical) on 512 deterministic IQ4_NL
rows; one mismatched byte aborts the run before real data is touched.

Output file layout (also what --merge re-checks at resume):
  GGUF header : shard1's raw KV bytes plus two added entries
                deepseek41.engram.encoding = "lloyd4_e5m3_32_r136"
                general.alignment          = 16384
                then 1008 resident tensor entries with offsets rebased
                to global 16 KiB piece bases, then the 2 Engram entries
                (type 24 = DS4_TENSOR_I8, dims (136, rows)).
  data region : shard1 data | shard2 data | shard3 data | shard4 data
                | blk.1 Engram table (converted) | blk.14 Engram table.
  Piece bases are 16 KiB aligned; blk.1's absolute offset therefore is
  page aligned and blk.14 follows it, page aligned too — satisfying the
  engine's "V4.1 Engram tables must follow the page-aligned main
  weights" rule.

Resume: the output is append-only, fsynced per chunk; rerunning
continues from the output size. Deleted shard sources are rebuilt from
<out>.merge-manifest.json + <out>.src-headers.raw (written before any
delete). The Engram companion is kept by default (--delete-engram).

Usage:
  deepseek41_ista_merge.py {--plan | --merge} DIR [--out FILE]
      [--delete-engram] [--no-delete-shards]
"""
import ctypes
import json
import math
import os
import struct
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GGUF_MAGIC = 0x46554747
GGUF_ALIGNMENT = 16384
CHUNK_BYTES = 64 << 20
EMIT_ROWS = 1 << 20                     # log cadence during conversion
ROWS_CHUNK = 1 << 18
RESERVE = 16 << 30                      # never dip below this free space
ENCODING_NAME = "lloyd4_e5m3_32_r136"
ROW144, ROW264, ROW136 = 144, 264, 136

SHARD_NAMES = [
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf",
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf",
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf",
]
TABLE_NAMES = ["blk.1.engram_embd.weight", "blk.14.engram_embd.weight"]
ENGRAM_FILE = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
OUT_DEFAULT = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-ds41.gguf"

SCALAR = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
          5: ("i", 4), 6: ("f", 4), 7: ("B", 1), 10: ("Q", 8), 11: ("q", 8),
          12: ("d", 8)}
T_STR, T_ARR, T_U32, T_U16, T_U64, T_BOOL = 8, 9, 4, 2, 10, 7


def die(msg):
    print(f"deepseek41_ista_merge: {msg}", file=sys.stderr)
    sys.exit(1)


def align(x, n):                        # n a power of two
    return (x + n - 1) & ~(n - 1)


def human(n):
    return float(n) / 1073741824.0


# --------------------------------------------------------------------- #
# float32 helpers (pure Python; struct pads quantify each op to f32)     #
# --------------------------------------------------------------------- #
def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _signbit(x):
    """C signbit() equivalent: True for -0.0 too."""
    return math.copysign(1.0, x) < 0


def fp16_to_float(h):
    hw = h & 0xFFFF
    sign = hw >> 15
    exp = (hw >> 10) & 31
    man = hw & 1023
    if exp == 31:
        v = math.nan if man else math.inf
    elif exp == 0:
        v = math.ldexp(float(man), -24) if man else 0.0
    else:
        v = math.ldexp(float(1024 + man), exp - 25)
    return -v if sign else v


# --------------------------------------------------------------------- #
# streaming GGUF cursor                                                 #
# --------------------------------------------------------------------- #
class Cursor:
    def __init__(self, fd, pos=0):
        self.fd, self.pos = fd, pos

    def u64(self):
        b = os.pread(self.fd, 8, self.pos)
        if len(b) != 8:
            die(f"short u64 read at {self.pos}")
        self.pos += 8
        return struct.unpack("<Q", b)[0]

    def u32(self):
        b = os.pread(self.fd, 4, self.pos)
        if len(b) != 4:
            die(f"short u32 read at {self.pos}")
        self.pos += 4
        return struct.unpack("<I", b)[0]

    def string(self):
        n = self.u64()
        b = os.pread(self.fd, n, self.pos)
        if len(b) != n:
            die(f"short string read ({len(b)} of {n}) at {self.pos}")
        self.pos += n
        return b.decode("utf-8", "replace")


def skip_value(c: Cursor, t):
    if t == T_STR:
        c.string()
        return
    if t == T_ARR:
        at = c.u32()
        cnt = c.u64()
        if at == T_STR:
            for _ in range(cnt):
                n = c.u64()
                c.pos += n      # skip the payload without reading it
            return
        fmt, sz = SCALAR[at]
        c.pos += sz * cnt
        return
    fmt, sz = SCALAR[t]
    if not sz:
        die(f"unknown kv value type {t}")
    c.pos += sz


# --------------------------------------------------------------------- #
# source parsing                                                        #
# --------------------------------------------------------------------- #
class Source:
    __slots__ = ("path", "fd", "reconstructed", "file_size", "llama_ds",
                 "n_kv", "n_tensors", "kv_start", "kv_end", "tensors",
                 "hdrlen", "kv_bytes_off", "piece_len")

    def __init__(self, path):
        self.path = path
        self.fd = None
        self.reconstructed = False

    def load_from_disk(self, need_piece_len=None):
        try:
            self.fd = os.open(self.path, os.O_RDONLY)
        except OSError as e:
            return False
        c = Cursor(self.fd)
        if c.u32() != GGUF_MAGIC:
            die(f"{self.path}: not a GGUF file")
        ver = c.u32()
        if ver not in (2, 3):
            die(f"{self.path}: GGUF v{ver} unsupported")
        # GGUF v2/v3 header: magic, version, n_tensors (u64), n_kv (u64)
        self.n_tensors, self.n_kv = c.u64(), c.u64()
        self.kv_start = c.pos
        for _ in range(self.n_kv):
            c.string()
            skip_value(c, c.u32())
        self.kv_end = c.pos
        self.tensors = []
        for _ in range(self.n_tensors):
            name = c.string()
            nd = c.u32()
            dims = [c.u64() for _ in range(nd)]
            ty = c.u32()
            rel = c.u64()
            self.tensors.append({"name": name, "ndim": nd, "dims": dims,
                                 "type": ty, "rel": rel})
        ds = align(c.pos, 32)
        self.llama_ds = ds
        self.hdrlen = ds
        st = os.fstat(self.fd)
        self.file_size = st.st_size
        self.piece_len = self.file_size - self.llama_ds
        if self.piece_len <= 0:
            die(f"{self.path}: no tensor data")
        if need_piece_len is not None and need_piece_len != self.piece_len:
            die(f"{self.path}: data length {self.piece_len} disagrees with "
                f"the manifest {need_piece_len}")
        return True

    def find(self, name):
        for t in self.tensors:
            if t["name"] == name:
                return t
        return None

    def min_rel(self):
        return min(t["rel"] for t in self.tensors)


def read_header_bytes(src: Source) -> bytes:
    return os.pread(src.fd, src.hdrlen, 0)


def kv_value_int(src: Source, key, which=0):
    """Find `key` in the parsed KV list and read numeric element which."""
    c = Cursor(src.fd, src.kv_start)
    for _ in range(src.n_kv):
        off = c.pos
        name = c.string()
        t = c.u32()
        if name == key:
            if t == T_ARR:
                at = c.u32()
                cnt = c.u64()
                if which >= cnt:
                    die(f"{key}: index {which} out of range ({cnt})")
                fmt, sz = SCALAR[at]
                raw = os.pread(src.fd, sz, c.pos + sz * which)
                return struct.unpack("<" + fmt, raw)[0]
            fmt, sz = SCALAR[t]
            raw = os.pread(src.fd, sz, c.pos)
            return struct.unpack("<" + fmt, raw)[0]
        skip_value(c, t)
    die(f"required metadata key missing: {key}")


# --------------------------------------------------------------------- #
# conversion kernel bindings + self-test                                #
# --------------------------------------------------------------------- #
CORE = None


def core_lib():
    global CORE
    if CORE is not None:
        return CORE
    path = os.path.join(REPO, "gguf-tools", "libds41ista.dylib")
    if not os.path.exists(path):
        die(f"{path} missing; first run:\n"
            f"  cc -O2 -o {path} "
            f"{REPO}/gguf-tools/deepseek41_ista_core.c {REPO}/ds4_engram.c -lm")
    CORE = ctypes.CDLL(path)
    CORE.ds41_ista_convert_chunk.restype = ctypes.c_int
    CORE.ds41_ista_convert_chunk.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64]
    enc = CORE.ds4_engram_encode_row136
    enc.restype = ctypes.c_bool
    enc.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                    ctypes.POINTER(ctypes.c_uint8)]
    return CORE


def parse_iq4nl_table():
    txt = open(os.path.join(REPO, "ds4_iq_tables.h"), "rb").read().decode()
    i = txt.index("kvalues_iq4nl[16]")
    j = txt.index("{", i)
    k = txt.index("}", j)
    vals = [int(x.strip().rstrip(",")) for x in txt[j + 1:k].split(",")
            if x.strip()]
    if len(vals) != 16:
        die("kvalues_iq4nl parse failed")
    return vals


def ref_row136(row144):
    """Independent pure-Python float32 reference of the conversion."""
    kvals = parse_iq4nl_table()
    v = [0.0] * 256
    for g in range(8):
        block = row144[18 * g:18 * g + 18]
        d = fp16_to_float(block[0] | (block[1] << 8))
        for j in range(16):
            v[32 * g + j] = f32(d * float(kvals[block[2 + j] & 0xF]))
            v[32 * g + 16 + j] = f32(d * float(kvals[block[2 + j] >> 4]))
    work = [0] * ROW264
    for g in range(8):
        gv = v[32 * g:32 * g + 32]
        amax = f32(max(abs(x) for x in gv)) if max(abs(x) for x in gv) else 0.0
        if amax > 0:
            m, e2 = math.frexp(amax)
            e2 = max(int(e2), -127)
            if e2 > 126:
                die("row scale exponent out of range")
            sc = e2 + 127
            for j in range(32):
                q = f32(abs(gv[j]) * math.ldexp(1.0, -e2))
                mm, ee = math.frexp(q)
                e0 = int(ee) + 6
                if e0 < 1:
                    k = _rint(q * 512.0)
                    code = int(k) if k <= 15 else 16
                else:
                    qp = f32(q * math.ldexp(1.0, 10 - e0))
                    man = _rint(qp) - 8
                    if man >= 8:
                        man, e0 = 0, e0 + 1
                    code = (e0 << 3) | int(man)
                if (code & 127) == 127 or not math.isfinite(
                        e4m3_ref_decode(code)):
                    die("invalid e4m3 code in reference")
                work[32 * g + j] = code | (128 if _signbit(gv[j]) else 0)
            work[256 + g] = sc
    return work, row136_encode(work)


def e4m3_ref_decode(code):
    exp, man = (code >> 3) & 15, code & 7
    return math.ldexp(float(8 + man), exp - 10) if exp \
        else math.ldexp(float(man), -9)


def _rint(x):
    """IEEE round-to-nearest-even on a nonnegative float."""
    if x >= 0:
        n = math.floor(x)
        frac = x - n
        if frac > 0.5:
            return n + 1
        if frac < 0.5:
            return n
        return n if n % 2 == 0 else n + 1
    raise ValueError


def row136_encode(work):
    """Call the fork's own encoder through the dylib (returns 136 B)."""
    lib = core_lib()
    src = (ctypes.c_uint8 * ROW264)()
    dst = (ctypes.c_uint8 * ROW136)()
    for i in range(ROW264):
        src[i] = work[i]
    if not lib.ds4_engram_encode_row136(src, dst):
        die(f"fork encoder rejected row (errno={ctypes.get_errno()})")
    return bytes(dst)


def selftest():
    state = 0x243F6A8885A308D3
    kvals = parse_iq4nl_table()
    WHITELIST = [0x3800, 0x3C00, 0x4000, 0x3400, 0x0000]
    rows = []
    for r in range(512):
        row = bytearray(ROW144)
        for g in range(8):
            state = (state * 6364136223846793005
                     + 1442695040888963407) & ((1 << 64) - 1)
            d16 = WHITELIST[state % len(WHITELIST)]
            block = bytearray(18)
            block[0] = d16 & 0xFF
            block[1] = (d16 >> 8) & 0xFF
            for i in range(16):
                state = (state * 25214903917 + 11) & ((1 << 48) - 1)
                block[2 + i] = state & 0xFF
            if r % 31 == 0:                 # exercise zero groups too
                block[2:18] = bytes(16)
            row[18 * g:18 * g + 18] = block
        rows.append(bytes(row))
    # convert via the C kernel
    buf = (ctypes.c_uint8 * (512 * ROW144))()
    dst_buf = (ctypes.c_uint8 * (512 * ROW136))()
    for i, b in enumerate(bytes(b"".join(rows))):
        buf[i] = b
    rc = core_lib().ds41_ista_convert_chunk(buf, dst_buf, 512)
    if rc != 0:
        die(f"selftest: C kernel returned {rc}")
    c_rows = bytes(dst_buf)
    # independent reference per row
    for r, row in enumerate(rows):
        work, ref136 = ref_row136(row)
        got = c_rows[r * ROW136:(r + 1) * ROW136]
        if got != bytes(ref136):
            die(f"selftest: row {r} disagree:\n  C:      {got.hex()}\n"
                f"  python: {bytes(ref136).hex()}")
    print("selftest: 512 deterministic rows match the Python reference "
          "byte-for-byte")


# --------------------------------------------------------------------- #
# plan                                                                  #
# --------------------------------------------------------------------- #
def build_plan(shards, engram, rows, out_path):
    header_t_bytes = 0
    for s in shards:
        for t in s.tensors:
            header_t_bytes += 8 + len(t["name"].encode()) + 4 \
                + t["ndim"] * 8 + 4 + 8
    header_t_bytes += 2 * (8 + len(TABLE_NAMES[0]) + 4 + 2 * 8 + 4 + 8)
    kv_bytes = shards[0].kv_end - shards[0].kv_start
    kv_bytes += 8 + len("deepseek41.engram.encoding") + 4 + 8 \
        + len(ENCODING_NAME)
    kv_bytes += 8 + len("general.alignment") + 4 + 4
    data_start = align(24 + kv_bytes + header_t_bytes, GGUF_ALIGNMENT)

    base = [0] * 6
    piece = [0] * 6
    b = data_start
    for s in range(4):
        base[s] = b
        piece[s] = shards[s].piece_len
        b = align(b + piece[s], GGUF_ALIGNMENT)
    base[4] = b                      # blk.1 table: must be page-aligned
    piece[4] = rows[0] * ROW136
    start_abs_0 = base[4]
    if start_abs_0 % GGUF_ALIGNMENT != 0:
        die(f"internal: blk.1 abs {start_abs_0} not page-aligned")
    b = align(b + piece[4], GGUF_ALIGNMENT)
    base[5] = b                      # blk.14 table starts right after tbl1
    piece[5] = rows[1] * ROW136
    start_abs_1 = base[5]
    final = base[5] + piece[5]
    return {"shards": shards, "engram": engram, "rows": rows,
            "out_path": out_path, "data_start": data_start, "base": base,
            "piece_len": piece, "start_abs": [start_abs_0, start_abs_1],
            "final": final}


def print_plan(p):
    print("=== deepseek41_ista_merge plan ===")
    print(f"out:             {p['out_path']}")
    print(f"data_start:      {p['data_start']}")
    for s, src in enumerate(p["shards"]):
        print(f"shard {s + 1}: {src.path} ({human(src.file_size):.2f} GiB, "
              f"{src.n_tensors} tensors, {src.n_kv} kv, "
              f"data {human(src.piece_len):.2f} GiB)"
              f"{' (RECONSTRUCTED)' if src.reconstructed else ''}")
    en = p["engram"]
    print(f"engram companion: {en.path} ({human(en.file_size):.2f} GiB, "
          f"{en.n_tensors} tensors)"
          f"{' (RECONSTRUCTED)' if en.reconstructed else ''}")
    for t in en.tensors:
        print(f"  {t['name']}: dims ({t['dims'][0]} x {t['dims'][1]}) "
              f"type {t['type']} rel {t['rel']}")
    for i in range(6):
        tag = "   <-- blk.1 table, page-aligned" if i == 4 else ""
        print(f"piece {i + 1}: base {p['base'][i]} + {p['piece_len'][i]} B{tag}")
    print(f"final size:      {p['final']} B ({human(p['final']):.2f} GiB)")
    print(f"engram blk.1 abs {p['start_abs'][0]} "
          f"(% 16384 = {p['start_abs'][0] % 16384}); "
          f"blk.14 abs {p['start_abs'][1]} "
          f"(% 16384 = {p['start_abs'][1] % 16384})")
    print(f"rows: blk.1 {p['rows'][0]}, blk.14 {p['rows'][1]}")
    print(f"free right now: {human(free_bytes(p['out_path'])):.2f} GiB")


def free_bytes(path):
    st = os.statvfs(os.path.dirname(path) or ".")
    # APFS reports f_bsize as 1 MiB (meaningless); f_frsize is the real
    # fundamental block size (4096) — trusting df-style numbers here.
    return st.f_bavail * st.f_frsize


def ensure_free(p, need):
    f = free_bytes(p["out_path"])
    if f < need + RESERVE:
        die(f"low disk: {human(f):.2f} GiB free, need "
            f"{human(need):.2f} GiB + {human(RESERVE):.2f} GiB reserve")


# --------------------------------------------------------------------- #
# merge                                                                 #
# --------------------------------------------------------------------- #
def serialize_header(p):
    shards = p["shards"]
    n_tensors = sum(s.n_tensors for s in shards) + 2
    n_kv = shards[0].n_kv + 2
    out = bytearray()
    out += struct.pack("<I", GGUF_MAGIC)
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", n_tensors)
    out += struct.pack("<Q", n_kv)
    raw_kv = os.pread(shards[0].fd, shards[0].kv_end - shards[0].kv_start,
                      shards[0].kv_start)
    out += raw_kv
    key = b"deepseek41.engram.encoding"
    out += struct.pack("<Q", len(key)) + key
    out += struct.pack("<I", T_STR)
    val = ENCODING_NAME.encode()
    out += struct.pack("<Q", len(val)) + val
    key2 = b"general.alignment"
    out += struct.pack("<Q", len(key2)) + key2
    out += struct.pack("<I", T_U32)
    out += struct.pack("<I", GGUF_ALIGNMENT)
    for s in range(4):
        src = shards[s]
        for t in src.tensors:
            nb = t["name"].encode()
            out += struct.pack("<Q", len(nb)) + nb
            out += struct.pack("<I", t["ndim"])
            for d in t["dims"]:
                out += struct.pack("<Q", d)
            out += struct.pack("<I", t["type"])
            rel = p["base"][s] + t["rel"] - p["data_start"]
            out += struct.pack("<Q", rel)
    for e in range(2):
        nb = TABLE_NAMES[e].encode()
        out += struct.pack("<Q", len(nb)) + nb
        out += struct.pack("<I", 2)
        out += struct.pack("<Q", ROW136)
        out += struct.pack("<Q", p["rows"][e])
        out += struct.pack("<I", 24)          # DS4_TENSOR_I8
        out += struct.pack("<Q", p["base"][4 + e] - p["data_start"])
    return bytes(out)


def write_header(p):
    hdr = serialize_header(p)
    if align(len(hdr), GGUF_ALIGNMENT) != p["data_start"]:
        die(f"header size {len(hdr)} does not back the planned data start "
            f"{p['data_start']}")
    hdr += b"\0" * (p["data_start"] - len(hdr))
    out = os.open(p["out_path"], os.O_RDWR | os.O_CREAT, 0o644)
    try:
        st = os.fstat(out)
        if st.st_size >= p["data_start"]:
            existing = os.pread(out, p["data_start"], 0)
            if existing != hdr:
                die("existing output header disagrees with the recomputed "
                    "layout; refusing to mix runs")
            return
        done = 0
        while done < len(hdr):
            done += os.pwrite(out, hdr[done:], done)
        os.fsync(out)
    finally:
        os.close(out)
    print(f"header written: {p['data_start']} B; data starts there")


def copy_piece(p, s, out_fd, t0):
    src = p["shards"][s]
    base, blen = p["base"][s], p["piece_len"][s]
    cur = os.fstat(out_fd).st_size
    progressed = max(0, cur - base)
    if progressed > blen:
        die(f"output extends past piece {s + 1}'s span")
    at = progressed - (progressed % CHUNK_BYTES)
    if src.reconstructed and at < blen:
        die(f"{src.path} was already merged+removed in an earlier run, "
            f"but only {progressed} of {blen} B of its data exists in the "
            "output; this resume cannot proceed from here")
    if at == 0:
        ensure_free(p, blen)
        tag = " (RECONSTRUCTED, data already merged)" if src.reconstructed else ""
        print(f"piece {s + 1}/6: {os.path.basename(src.path)} → "
              f"{human(blen):.2f} GiB at {base}{tag}")
    while at < blen:
        n = min(CHUNK_BYTES, blen - at)
        chunk = bytearray(n)
        done = 0
        while done < n:
            g = os.pread(src.fd, n - done, src.llama_ds + at + done)
            if not g:
                die(f"short read of {src.path} at {src.llama_ds + at + done}")
            chunk[done:done + len(g)] = g
            done += len(g)
        done = 0
        while done < n:
            done += os.pwrite(out_fd, chunk[done:], base + at + done)
        os.fsync(out_fd)
        if ((at + n) % (4 << 30)) < CHUNK_BYTES:
            print(f"  piece {s + 1}/6: {human(at + n):6.2f} / "
                  f"{human(blen):6.2f} GiB, {time.time() - t0:6.0f}s, "
                  f"free {human(free_bytes(p['out_path'])):6.2f} GiB",
                  flush=True)
        at += n
    print(f"piece {s + 1}/6 done", flush=True)


def convert_piece(p, which, out_fd, t0):
    en = p["engram"]
    t = en.find(TABLE_NAMES[which])
    if t is None:
        die(f"companion lacks {TABLE_NAMES[which]}")
    if t["dims"][0] != 256 or t["type"] != 20 or t["ndim"] != 2:
        die(f"companion {TABLE_NAMES[which]}: expected ndim 2 dims "
            f"(256, rows) type 20; got ndim {t['ndim']} dims {t['dims']} "
            f"type {t['type']}")
    rows = p["rows"][which]
    src_abs = en.llama_ds + t["rel"]
    base = p["base"][4 + which]
    cur = os.fstat(out_fd).st_size
    progressed = max(0, cur - base)
    rows_done0 = progressed // ROW136
    rows_done0 -= rows_done0 % ROWS_CHUNK
    if en.reconstructed and rows_done0 < rows:
        die("companion is already absent but its converted table "
            f"{TABLE_NAMES[which]} is incomplete in the output; "
            "the resume cannot proceed from here")
    if rows_done0 == 0:
        ensure_free(p, rows * ROW136)
        print(f"piece {5 + which}/6: {TABLE_NAMES[which]}: {rows} rows "
              f"(→ {human(rows * ROW136):.2f} GiB at {base})")
    src_buf = (ctypes.c_uint8 * (ROWS_CHUNK * ROW144))()
    dst_buf = (ctypes.c_uint8 * (ROWS_CHUNK * ROW136))()
    for r0 in range(rows_done0, rows, ROWS_CHUNK):
        n_rows = min(ROWS_CHUNK, rows - r0)
        need = n_rows * ROW144
        done = 0
        while done < need:
            g = os.pread(en.fd, need - done, src_abs + r0 * ROW144 + done)
            if not g:
                die(f"short read of {en.path} row {r0 + done // ROW144}")
            ctypes.memmove(ctypes.byref(src_buf, done), g, len(g))
            done += len(g)
        rc = core_lib().ds41_ista_convert_chunk(src_buf, dst_buf, n_rows)
        if rc != 0:
            die(f"conversion kernel error {rc} at table "
                f"{TABLE_NAMES[which]} row {r0}")
        w = n_rows * ROW136
        blob = bytes(memoryview(dst_buf)[:w])
        done = 0
        while done < w:
            done += os.pwrite(out_fd, blob[done:],
                              base + r0 * ROW136 + done)
        os.fsync(out_fd)
        if (r0 + n_rows) % (1 << 22) < ROWS_CHUNK:
            print(f"  piece {5 + which}/6: {r0 + n_rows}/{rows} rows "
                  f"({100.0 * (r0 + n_rows) / rows:.1f}%), "
                  f"{time.time() - t0:6.0f}s, "
                  f"free {human(free_bytes(p['out_path'])):6.2f} GiB",
                  flush=True)
    print(f"piece {5 + which}/6 done", flush=True)


def write_manifest(p, raws):
    man_path = p["out_path"] + ".merge-manifest.json"
    raw_path = p["out_path"] + ".src-headers.raw"
    recs = []
    offset = 0
    for s, src in enumerate([*p["shards"], p["engram"]]):
        recs.append({
            "path": src.path, "llama_ds": src.llama_ds,
            "n_kv": src.n_kv, "n_tensors": src.n_tensors,
            "kv_start": src.kv_start, "kv_end": src.kv_end,
            "piece_len": src.piece_len,
            "hdr_off": offset, "hdr_len": len(raws[s]),
            "tensors": src.tensors,
        })
        offset += len(raws[s])
    man = {"version": 1, "out": p["out_path"],
           "data_start": p["data_start"], "base": p["base"],
           "piece_len": p["piece_len"], "start_abs": p["start_abs"],
           "rows": list(p["rows"]), "final": p["final"],
           "shards": recs[0:4], "engram": recs[4]}
    with open(man_path, "w") as f:
        json.dump(man, f)
    with open(raw_path, "wb") as f:
        for r in raws:
            f.write(r)
    print(f"manifest: {man_path} ({offset} B of source headers archived)")


def load_manifest(out_path):
    man_path = out_path + ".merge-manifest.json"
    try:
        with open(man_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def merge(dirpath, out_path, delete_engram=True, delete_shards=True):
    if not out_path:
        out_path = os.path.join(dirpath, OUT_DEFAULT)
    raw_path = out_path + ".src-headers.raw"
    man = load_manifest(out_path)

    sources = []
    raws = []
    for s, name in enumerate(SHARD_NAMES):
        path = os.path.join(dirpath, name)
        src = Source(path)
        loaded = src.load_from_disk()
        if not loaded:
            if man is None:
                die(f"cannot open {path} and no manifest exists")
            rec = man["shards"][s]
            if os.path.basename(rec["path"]) != name:
                die("manifest path mismatch")
            src = Source(path)
            src.reconstructed = True
            src.path = rec["path"]
            src.llama_ds = rec["llama_ds"]
            src.n_kv = rec["n_kv"]
            src.n_tensors = rec["n_tensors"]
            src.kv_start, src.kv_end = rec["kv_start"], rec["kv_end"]
            src.tensors = rec["tensors"]
            src.piece_len = rec["piece_len"]
            src.kv_bytes_off = rec["hdr_off"]
            src.hdrlen = rec["hdr_len"]
            src.fd = os.open(raw_path, os.O_RDONLY)  # only for KV reads
        sources.append(src)

    en_path = os.path.join(dirpath, ENGRAM_FILE)
    en = Source(en_path)
    if not en.load_from_disk():
        if man is None:
            die(f"cannot open {en_path} and no manifest exists")
        rec = man["engram"]
        en.reconstructed = True
        en.path = rec["path"]
        en.llama_ds = rec["llama_ds"]
        en.n_tensors = rec["n_tensors"]
        en.kv_start, en.kv_end = rec["kv_start"], rec["kv_end"]
        en.tensors = rec["tensors"]
        en.piece_len = rec["piece_len"]
        en.kv_bytes_off = rec["hdr_off"]
        en.hdrlen = rec["hdr_len"]
        en.fd = os.open(raw_path, os.O_RDONLY)

    rows = [kv_value_int(sources[0], "deepseek41.engram.num_embeddings", 0),
            kv_value_int(sources[0], "deepseek41.engram.num_embeddings", 1)]
    split_count = kv_value_int(sources[0], "split.tensors.count", 0)
    total_tensors = sum(s.n_tensors for s in sources)
    if total_tensors != split_count:
        die(f"tensor counts disagree: shards carry {total_tensors}, "
            f"split.tensors.count says {split_count}")
    for s, src in enumerate(sources):
        if src.min_rel() != 0:
            die(f"shard {s + 1}: lowest tensor offset is {src.min_rel()}, "
                "expected 0 — llama split layout assumption broken")
    if sources[0].find("engram_token_map.weight") is None:
        die("shard1 lacks engram_token_map.weight")
    for e, name in enumerate(TABLE_NAMES):
        t = en.find(name)
        if t is None:
            die(f"companion lacks {name}")
        if t["dims"][1] != rows[e]:
            die(f"companion {name}: rows {t['dims'][1]} != metadata "
                f"{rows[e]}")

    p = build_plan(sources, en, rows, out_path)
    if p["start_abs"][0] % 4096:
        die("internal: engram tbl start not page aligned")
    if p["start_abs"][1] < p["start_abs"][0] + p["piece_len"][4]:
        die("internal: engram tables overlap")
    if ROW136 != 136:
        die("unexpected row bytes")
    print_plan(p)

    t0 = time.time()
    selftest()                # deterministic cross-check; aborts on mismatch

    if man is None:
        raws = [read_header_bytes(s) for s in [*sources, en]]
        write_manifest(p, raws)
    else:
        print(f"manifest exists: {out_path}.merge-manifest.json")

    write_header(p)
    out_fd = os.open(out_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        for s in range(4):
            copy_piece(p, s, out_fd, t0)
            if delete_shards and not sources[s].reconstructed \
                    and os.path.exists(sources[s].path):
                os.unlink(sources[s].path)
                print(f"  ({os.path.basename(sources[s].path)} removed)")
        for e in range(2):
            convert_piece(p, e, out_fd, t0)
        if delete_engram and os.path.exists(en.path):
            if en.reconstructed:
                print("  (companion already absent)")
            else:
                os.unlink(en.path)
                print(f"  ({os.path.basename(en.path)} removed)")
        else:
            print(f"  (companion kept: {en.path})")
        sz = os.fstat(out_fd).st_size
        if sz != p["final"]:
            die(f"final size {sz} != planned {p['final']}")
    finally:
        os.close(out_fd)

    # final integrity: reparse the finished GGUF header
    fd = os.open(out_path, os.O_RDONLY)
    c = Cursor(fd)
    if c.u32() != GGUF_MAGIC:
        die("final file lost its GGUF magic")
    ver = c.u32()
    n_tensors, n_kv = c.u64(), c.u64()
    want_kv = sources[0].n_kv + 2
    if n_tensors != sum(s.n_tensors for s in sources) + 2 or n_kv != want_kv:
        die(f"final header shape wrong: tensors {n_tensors}, kv {n_kv} "
            f"(wanted {total_tensors + 2}/{want_kv})")
    os.close(fd)
    print(f"done: {out_path} ({human(p['final']):.2f} GiB), "
          f"{time.time() - t0:.0f}s total")
    print("smoke the result with:")
    print(f"  ./ds4 -m {out_path} --metal --ssd-streaming ...")


def plan(dirpath, out_path):
    if not out_path:
        out_path = os.path.join(dirpath, OUT_DEFAULT)
    sources = []
    for name in SHARD_NAMES:
        src = Source(os.path.join(dirpath, name))
        if not src.load_from_disk():
            die(f"cannot open {src.path} (plan requires every present "
                f"file)")
        sources.append(src)
    en = Source(os.path.join(dirpath, ENGRAM_FILE))
    if not en.load_from_disk():
        die("cannot open the engram companion")
    rows = [kv_value_int(sources[0], "deepseek41.engram.num_embeddings", 0),
            kv_value_int(sources[0], "deepseek41.engram.num_embeddings", 1)]
    split_count = kv_value_int(sources[0], "split.tensors.count", 0)
    total_tensors = sum(s.n_tensors for s in sources)
    if total_tensors != split_count:
        die(f"tensor counts disagree: shards carry {total_tensors}, "
            f"split.tensors.count says {split_count}")
    for s, src in enumerate(sources):
        if src.min_rel() != 0:
            die(f"shard {s + 1}: lowest tensor offset is {src.min_rel()}, "
                "expected 0 — llama split layout assumption broken")
    for e, name in enumerate(TABLE_NAMES):
        t = en.find(name)
        if t is None:
            die(f"companion lacks {name}")
        if t["dims"][1] != rows[e]:
            die(f"companion {name}: rows {t['dims'][1]} != metadata "
                f"{rows[e]}")
    p = build_plan(sources, en, rows, out_path)
    print_plan(p)


def main(argv):
    flags = {"--plan", "--merge", "--delete-engram", "--no-delete-shards"}
    positional = []
    out_path = None
    i = 0
    do_plan = do_merge = False
    delete_engram, delete_shards = False, True   # keep companion by default
    while i < len(argv):
        a = argv[i]
        if a == "--plan":
            do_plan = True
        elif a == "--merge":
            do_merge = True
        elif a == "--delete-engram":
            delete_engram = True
        elif a == "--no-delete-engram":
            delete_engram = False
        elif a == "--no-delete-shards":
            delete_shards = False
        elif a == "--out":
            i += 1
            if i >= len(argv):
                die("--out needs a value")
            out_path = argv[i]
        elif a.startswith("-"):
            die(f"unknown option {a}")
        else:
            positional.append(a)
        i += 1
    if (do_plan == do_merge) or len(positional) != 1:
        die("usage: deepseek41_ista_merge.py {--plan | --merge} DIR "
            "[--out FILE] [--delete-engram | --no-delete-engram] "
            "[--no-delete-shards]")
    core_lib()                    # abort early with a clear message
    if do_plan:
        plan(positional[0], out_path)
    else:
        merge(positional[0], out_path, delete_engram, delete_shards)


if __name__ == "__main__":
    main(sys.argv[1:])
