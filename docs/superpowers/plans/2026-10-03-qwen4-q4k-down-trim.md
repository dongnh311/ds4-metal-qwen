# Trimmed Q4_K Down Rows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a lossless unc48L file whose Q4_K routed down rows drop the 640→768 padding (368 B rows instead of
432 B), and ds4 support for it with output byte-identical to PROD.

**Architecture:**
- A gguf_lite converter cuts each padded row after `16 + (F % 256) / 2` bytes of its last block.
- ds4.c learns the short-final-block row size: parse, stream planner, validator and CPU reference.
- The Metal host computes down row bytes through one helper that a per-load flag switches.
- Kernels are unchanged: they already skip values from F on.

**Tech Stack:** C (ds4.c), Objective-C/Metal host (ds4_metal.m), Python 3 + numpy (tools), zsh measurement
scripts.

**Spec:** `docs/superpowers/specs/2026-10-03-qwen4-q4k-down-trim-design.md`

## Global Constraints

- No push, merge, deploy or HF upload without the user's OK. The branch stays local.
- Never use bare `git stash`. Never remove the kv-grow worktree. Never `kill -9` a Metal process.
- Code, docs and commits in English. Commits end with the two attribution lines (Co-Authored-By, Claude-Session).
- Do not touch DS41F's `DeepSeek-V4.1-Flash-Q2.gguf` or its `.eq4-*` side files.
- GPU work only inside announced windows:
  - Peers: DS41F `uds:/tmp/cc-socks/45358.sock`, Ornith `uds:/tmp/cc-socks/34890.sock`, Ai-Gateway-CodeTab
    `uds:/tmp/cc-socks/28613.sock`, Ai-Gateway `uds:/tmp/cc-socks/73364.sock`.
  - A window is at most 60 min, ends with "done" plus backend_ok, and the gateway is restored by `hold-gateway.sh`.
  - Never during another session's production window.
- PROD A/B speed runs control page-cache warmth: each arm twice back to back, compare the second runs.
- Models live in `/Users/dongnh/orca/workspaces/ds4-metal-data/gguf`. Git holds source only.
- PROD file:
  `/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf`.
- Trimmed file:
  `/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownTrim-DenseQ4Kselimat-MTP.gguf`.
- PROD env and flags:
  - env `DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0 DS4_QWEN4_KV_GROW=1`;
  - flags `--metal --ple <PLE> --prefill-chunk 2048 --ssd-streaming --ssd-streaming-cache-experts 6GB`.

## Review Focus

1. **A layer mix: some Q4_K down tensors trimmed, some padded, in one file.** Expected: refused at load, not a
   silent wrong stride. Pinned in Task 3 (`check_uniform_layout`).
2. **The MTP layer's Q8_0 down `[640, 2560, 512]` next to trimmed Q4_K layers.** Expected: accepted, since only
   Q4_K tensors count toward uniformity. Pinned in Task 3.
3. **A trimmed file on a non-Metal GPU build.** Expected: a clear refusal at load. Covered by the
   `#elif defined(DS4_HAS_QWEN4_GPU)` branch in Task 3 (compile-only on this Mac; noted for the reviewer).
4. **The flag left on, then a padded model loaded in the same process.** Expected: the loader resets the flag on
   every qwen4 load. Pinned in Task 2 (the kernel test turns it off and on) and Task 3 (the validator always calls
   the setter).
5. **The converter run on a file that is already trimmed, or on a non-qwen4 file.** Expected: refused, nothing
   written. Pinned in Task 1.

---

### Task 0: Baseline binaries (no GPU)

**Files:** none in git. Output: `/Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/base/{ds4,ds4-bench,ds4-server}`.

- [ ] **Step 1: Build develop ef4d897c and keep the binaries**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/kv-grow
git rev-parse HEAD   # expect d5640860 (spec) whose parent is ef4d897c; no code change yet
make -j8 ds4 ds4-bench ds4-server > /Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/build-base.log 2>&1
mkdir -p /Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/base
cp ds4 ds4-bench ds4-server /Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/base/
```

Expected: build exit 0. The three binaries are copied. The commit adds docs only, so these are develop's
binaries.

---

### Task 1: gguf_lite short final block + converter

**Files:**
- Modify: `tools/gguf_lite.py` (`nbytes`)
- Create: `tools/qwen4_trim_down_pad.py`
- Test: `tools/test_gguf_lite.py` (one new test), `tools/test_qwen4_trim_down_pad.py` (new)

**Interfaces:**
- Produces: `gguf_lite.q4k_row_bytes(n) -> int`, which returns 0 when n % 64 != 0.
- Produces: `qwen4_trim_down_pad.plan(src) -> (tensors, trimmed)`.
- Produces: `qwen4_trim_down_pad.trim_rows(buf: bytes, in_row: int, out_row: int) -> bytes`.
- CLI: `python3 tools/qwen4_trim_down_pad.py IN.gguf OUT.gguf`, which writes OUT and OUT.json.

- [ ] **Step 1: Write the failing gguf_lite test** (append to `GgufLiteTest`)

```python
    def test_q4k_short_final_block(self):
        self.assertEqual(g.q4k_row_bytes(768), 432)
        self.assertEqual(g.q4k_row_bytes(640), 368)
        self.assertEqual(g.q4k_row_bytes(320), 192)
        self.assertEqual(g.q4k_row_bytes(704), 400)
        self.assertEqual(g.q4k_row_bytes(672), 0)
        self.assertEqual(g.nbytes(12, [640, 3, 2]), 368 * 6)
        self.assertEqual(g.nbytes(12, [768, 3]), 432 * 3)
        with self.assertRaises(ValueError):
            g.nbytes(12, [672, 3])
```

- [ ] **Step 2: Run it**

Run: `cd tools && python3 -m unittest test_gguf_lite -v`
Expected: FAIL. `AttributeError: module 'gguf_lite' has no attribute 'q4k_row_bytes'`.

- [ ] **Step 3: Implement in `tools/gguf_lite.py`** (above `nbytes`, and a branch at the top of `nbytes`)

```python
def q4k_row_bytes(n):
    """Bytes of a Q4_K row of n values. A row that ends inside a super-block (n % 256 != 0) keeps that block's
    16-byte header (d, dmin, 12 scale bytes) and only the 32-byte qs chunks of its real values; 0 if n is not
    a multiple of 64 (a chunk holds 64 values)."""
    if n % 64:
        return 0
    return n // 256 * 144 + (16 + n % 256 // 2 if n % 256 else 0)
```

In `nbytes`, before the `elems, size = BLOCK[ttype]` line:

```python
    if ttype == 12 and dims[0] % 256:
        row = q4k_row_bytes(dims[0])
        if not row:
            raise ValueError("Q4_K row of %d is not a multiple of 64" % dims[0])
        for d in dims[1:]:
            row *= d
        return row
```

- [ ] **Step 4: Run it**

Run: `cd tools && python3 -m unittest test_gguf_lite -v`
Expected: all tests pass.

- [ ] **Step 5: Write the failing converter tests** in `tools/test_qwen4_trim_down_pad.py`

```python
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import gguf_lite as g
import qwen4_trim_down_pad as tr

HERE = pathlib.Path(__file__).resolve().parent


def padded_rows(rows, seed):
    """rows x 432 bytes: 3 Q4_K blocks per row, every byte distinct-ish (padding nonzero too)."""
    return bytes((i * 7 + seed) & 0xFF for i in range(rows * 432))


class TrimTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.dir.name)
        self.kv = {"general.architecture": (g.T_STR, "qwen4exp"),
                   "qwen4exp.expert_feed_forward_length": (g.T_U32, 640)}
        # 2 experts x 3 output rows per expert: dims [768, 3, 2]
        self.down = padded_rows(6, 1)
        self.other = bytes(range(64))
        self.tensors = [
            {"name": "blk.0.ffn_down_exps.weight", "dims": [768, 3, 2], "type": 12, "nbytes": 6 * 432,
             "data": self.down},
            {"name": "blk.0.attn_norm.weight", "dims": [16], "type": 0, "nbytes": 64, "data": self.other},
            {"name": "blk.1.ffn_down_exps.weight", "dims": [640, 3, 2], "type": 8, "nbytes": 6 * 20 * 34,
             "data": bytes(6 * 20 * 34)},   # MTP-like Q8_0 down: copied as is
        ]
        g.write(self.p / "in.gguf", self.kv, self.tensors, 32)

    def tearDown(self):
        self.dir.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, str(HERE / "qwen4_trim_down_pad.py"), *map(str, args)],
                              capture_output=True, text=True)

    def test_trim_rows(self):
        self.assertEqual(tr.trim_rows(self.down, 432, 368),
                         b"".join(self.down[r * 432:r * 432 + 368] for r in range(6)))

    def test_converts_q4k_down_only(self):
        r = self.run_tool(self.p / "in.gguf", self.p / "out.gguf")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = g.Reader(self.p / "out.gguf")
        by = {t["name"]: t for t in out.tensors}
        down = by["blk.0.ffn_down_exps.weight"]
        self.assertEqual((down["dims"], down["type"], down["nbytes"]), ([640, 3, 2], 12, 6 * 368))
        with open(self.p / "out.gguf", "rb") as f:
            f.seek(down["abs"])
            self.assertEqual(f.read(down["nbytes"]), tr.trim_rows(self.down, 432, 368))
            f.seek(by["blk.0.attn_norm.weight"]["abs"])
            self.assertEqual(f.read(64), self.other)
        self.assertEqual(by["blk.1.ffn_down_exps.weight"]["dims"], [640, 3, 2])
        self.assertEqual(out.kv, self.kv)

    def test_manifest(self):
        self.run_tool(self.p / "in.gguf", self.p / "out.gguf")
        m = json.loads((self.p / "out.gguf.json").read_text())
        t = m["tensors"]["blk.0.ffn_down_exps.weight"]
        self.assertEqual((t["bytes"], t["trimmed_from"]), (6 * 368, [768, 3, 2]))
        self.assertIsNone(m["tensors"]["blk.0.attn_norm.weight"]["trimmed_from"])
        self.assertEqual(m["bytes"], (self.p / "out.gguf").stat().st_size)

    def test_already_trimmed_refused(self):
        self.run_tool(self.p / "in.gguf", self.p / "out.gguf")
        r = self.run_tool(self.p / "out.gguf", self.p / "again.gguf")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nothing to trim", r.stderr)
        self.assertFalse((self.p / "again.gguf").exists())

    def test_non_qwen4_refused(self):
        kv = dict(self.kv, **{"general.architecture": (g.T_STR, "llama")})
        g.write(self.p / "llama.gguf", kv, self.tensors, 32)
        r = self.run_tool(self.p / "llama.gguf", self.p / "o.gguf")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("qwen4exp", r.stderr)

    def test_bad_ff_refused(self):
        kv = dict(self.kv, **{"qwen4exp.expert_feed_forward_length": (g.T_U32, 672)})
        g.write(self.p / "bad.gguf", kv, self.tensors, 32)
        r = self.run_tool(self.p / "bad.gguf", self.p / "o.gguf")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("multiple of 64", r.stderr)

    def test_usage(self):
        r = self.run_tool()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("usage", r.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Run them**

Run: `cd tools && python3 -m unittest test_qwen4_trim_down_pad -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qwen4_trim_down_pad'`.

- [ ] **Step 7: Implement `tools/qwen4_trim_down_pad.py`**

```python
"""Rewrite a qwen4exp GGUF whose routed down experts are Q4_K rows padded to whole 256-value super-blocks
(dims [roundup256(F), n_embd, n_expert]) as trimmed rows of F values: each row keeps its first
gguf_lite.q4k_row_bytes(F) bytes (whole blocks, then the last block's 16-byte header and the qs chunks of the real
values). The values the kernels read are unchanged; every other tensor and all metadata are copied as is.

usage: python3 tools/qwen4_trim_down_pad.py IN.gguf OUT.gguf   (writes OUT and OUT.json)"""
import json
import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_lite as g  # noqa: E402

Q4_K = 12
DOWN = re.compile(r"blk\.\d+\.ffn_down_exps\.weight$")


def trim_rows(buf, in_row, out_row):
    rows = np.frombuffer(buf, dtype=np.uint8).reshape(-1, in_row)
    return np.ascontiguousarray(rows[:, :out_row]).tobytes()


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
            with open(src.path, "rb") as f:
                f.seek(t["abs"])
                data = trim_rows(f.read(t["nbytes"]), in_row, out_row)
            out.update(dims=[ff] + t["dims"][1:], data=data, nbytes=len(data))
            del out["src"]
            trimmed[t["name"]] = t["dims"]
        tensors.append(out)
    return tensors, trimmed


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    in_path, out_path = argv[1], argv[2]
    src = g.Reader(in_path)
    tensors, trimmed = plan(src)
    if not trimmed:
        sys.exit("%s: no padded Q4_K routed down tensor; nothing to trim, nothing written" % in_path)
    tmp = out_path + ".partial"
    try:
        shas = g.write(tmp, src.kv, tensors, src.alignment)
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
```

Note: `g.write` holds each `data` blob in memory until the write ends. The PROD file has 48 trimmed tensors of
460 MiB each, about 22 GiB in total. That is too much next to a 52 GB page cache.

**Ruling (planned):** `plan` stores a lazy source instead of `data`.
- The source is `{"trim": (path, abs, in_row, out_row)}`.
- `gguf_lite.write` gets a small extension: when a tensor has `"trim"`, it streams the input in row-aligned
  64 MiB chunks through `trim_rows`.
- Implement that instead of `data` in Step 7. The tests above do not change, because they read the output
  bytes.

The extension in `gguf_lite.write`'s per-tensor branch:

```python
            elif "trim" in t:
                src, at, in_row, out_row = t["trim"]
                rows_left = t["nbytes"] // out_row
                step = max(1, CHUNK // in_row)
                with open(src, "rb") as f:
                    f.seek(at)
                    while rows_left:
                        n = min(step, rows_left)
                        buf = f.read(n * in_row)
                        if len(buf) != n * in_row:
                            raise ValueError("%s: source %s ends early" % (t["name"], src))
                        rows = memoryview(buf).cast("B")
                        for r in range(n):
                            piece = rows[r * in_row:r * in_row + out_row]
                            out.write(piece)
                            h.update(piece)
                        rows_left -= n
```

The extension uses a pure-Python loop so that gguf_lite does not depend on numpy.

**Ruling:** if this runs slower than 300 MB/s on PROD (48 × 2560 × 512 = 63M rows), use a numpy-optional
fast path in the converter instead: pass precomputed `data` per tensor, one tensor at a time, through a
generator. Measure on the real file in Task 4.

`plan` then sets `out.update(dims=..., nbytes=t["dims"][1] * t["dims"][2] * out_row, trim=(src.path, t["abs"],
in_row, out_row))`, deletes `src`, and does not read the data.

- [ ] **Step 8: Run all tool tests**

Run: `cd tools && python3 -m unittest test_gguf_lite test_qwen4_trim_down_pad test_ista_hc_to_f16 -v`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add tools/gguf_lite.py tools/qwen4_trim_down_pad.py tools/test_gguf_lite.py tools/test_qwen4_trim_down_pad.py
git commit -m "tools: qwen4_trim_down_pad, lossless trim of padded Q4_K routed down rows"
```

---

### Task 2: Metal host down row bytes + flag + kernel test

**Files:**
- Modify: `ds4_gpu.h` (declare the setter near `ds4_gpu_qwen4_set_rope`)
- Modify: `ds4_metal.m`:
  - the flag and helper next to `qwen4_expert_row_bytes` (about line 50312);
  - five sites at about lines 52225, 52409, 52813, 54280 and 54844.
- Test: `tests/test_qwen4_kernels.c` (new `test_down_trim`, called in the `__APPLE__` section of main)

**Interfaces:**
- Produces: `void ds4_gpu_qwen4_set_down_trimmed(bool on);` (Metal only).
- Produces: `static uint32_t qwen4_down_row_bytes(uint32_t type, uint32_t ff_dim)` in ds4_metal.m.

- [ ] **Step 1: Write the failing kernel test** (in `tests/test_qwen4_kernels.c`, inside an `#ifdef __APPLE__`
  block before `main`)

```c
/* Trimmed Q4_K down rows (docs/superpowers/specs/2026-10-03-qwen4-q4k-down-trim-design.md): a padded arena and
 * its trimmed copy (each row cut after 16 + (F % 256) / 2 bytes of its last block) must give bit-identical
 * down outputs through every decode geometry and the prefill GEMM. */
static void test_down_trim(arena_t *a, uint32_t F, uint32_t T) {
    const uint32_t NE = 8, slots = 6, E = 256, DF = (F + 255u) / 256u * 256u;
    const uint32_t in_row = DF / 256u * 144u, out_row = F / 256u * 144u + 16u + (F % 256u) / 2u;
    double *dw;
    const uint64_t pad_off = arena_q4_K(a, (uint64_t)NE * E, DF, &dw, 0.05f);
    free(dw);
    const uint64_t trim_off = arena_alloc(a, (uint64_t)NE * E * out_row);
    for (uint64_t r = 0; r < (uint64_t)NE * E; r++)
        memcpy(a->base + trim_off + r * out_row, a->base + pad_off + r * in_row, out_row);
    const uint32_t n_out = slots;
    float *mid = rand_vec((uint64_t)T * n_out * F, 1.0f);
    int32_t *sel = malloc((uint64_t)T * slots * 4);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < slots; s++) sel[t * slots + s] = (int32_t)((t * 5u + s * 3u) % NE);
    ds4_gpu_tensor *gmid = upload(mid, (uint64_t)T * n_out * F);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * slots * 4);
    require_ok(ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * slots * 4), "trim sel write");
    ds4_gpu_tensor *gpart = upload(NULL, (uint64_t)T * n_out * E);
    const uint64_t np = (uint64_t)T * n_out * E;
    float *ref = malloc(np * sizeof(float)), *got = malloc(np * sizeof(float));
    const char *mr[] = {"0", "2", "4"};
    for (uint32_t spec = 0; spec < 2u; spec++) {
        setenv("DS4_QWEN4_MOE_MV_SPECIALIZE", spec ? "1" : "0", 1);
        for (uint32_t m = 0; m < 3u; m++) {
            setenv("DS4_QWEN4_MOE_MR_DOWN", mr[m], 1);
            ds4_gpu_qwen4_set_down_trimmed(false);
            require_ok(ds4_gpu_qwen4_moe_down_tensor(gpart, gmid, gsel, a->base, a->size, pad_off, 12u, NE, T, slots,
                                                     F, E, 0, UINT32_MAX), "padded down");
            require_ok(ds4_gpu_tensor_read(gpart, 0, ref, np * sizeof(float)), "padded down read");
            ds4_gpu_qwen4_set_down_trimmed(true);
            require_ok(ds4_gpu_qwen4_moe_down_tensor(gpart, gmid, gsel, a->base, a->size, trim_off, 12u, NE, T, slots,
                                                     F, E, 0, UINT32_MAX), "trimmed down");
            require_ok(ds4_gpu_tensor_read(gpart, 0, got, np * sizeof(float)), "trimmed down read");
            check_exact_f32("trimmed Q4_K down (decode rows)", got, ref, np);
        }
    }
    unsetenv("DS4_QWEN4_MOE_MV_SPECIALIZE");
    unsetenv("DS4_QWEN4_MOE_MR_DOWN");
    /* the padded stride with the flag off again: the flag must not stick */
    ds4_gpu_qwen4_set_down_trimmed(false);
    require_ok(ds4_gpu_qwen4_moe_down_tensor(gpart, gmid, gsel, a->base, a->size, pad_off, 12u, NE, T, slots,
                                             F, E, 0, UINT32_MAX), "padded down again");
    require_ok(ds4_gpu_tensor_read(gpart, 0, got, np * sizeof(float)), "padded again read");
    check_exact_f32("padded Q4_K down after the flag is cleared", got, ref, np);
    free(ref); free(got); free(mid); free(sel);
    ds4_gpu_tensor_free(gmid); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gpart);
    printf("  trimmed Q4_K down F=%u T=%u: decode rows (MR 0/2/4, generic/specialized) byte-exact vs padded\n", F, T);
}
```

Prefill GEMM part (same function, before the frees). Build lists and counts the way the existing
`moe_mm_down_tensor` test does around `tests/test_qwen4_kernels.c:1791-1815`:
- Copy that block's list/count construction verbatim, with `NE`, `T`, `slots`, `n_out = slots`, `F`, `E`.
- Call `ds4_gpu_qwen4_moe_mm_down_tensor` twice: padded with the flag off, trimmed with the flag on.
- Compare with `check_exact_f32("trimmed Q4_K down (prefill GEMM)", ...)`.
- Run it only when `T >= 64`, the mm path's minimum. Main calls it with T = 1, 2, 9 and 128.

Calls in `main`'s `__APPLE__` section, next to the `test_moe_types(&arena, 8, 6, 2560, 640, ...)` lines:

```c
        test_down_trim(&arena, 640, 1);
        test_down_trim(&arena, 640, 2);
        test_down_trim(&arena, 640, 9);
        test_down_trim(&arena, 640, 128);
        test_down_trim(&arena, 320, 2);
        test_down_trim(&arena, 320, 128);
```

- [ ] **Step 2: Run it**

Run: `make tests/test_qwen4_kernels 2>&1 | tail -3`
Expected: compile FAIL, with an implicit declaration or undefined symbol `ds4_gpu_qwen4_set_down_trimmed`.

- [ ] **Step 3: Implement**

In `ds4_gpu.h`, next to the `ds4_gpu_qwen4_set_rope` declaration:

```c
/* Routed Q4_K down rows of the loaded qwen4 model are trimmed (ff_dim values, the last super-block cut after
 * its header and real qs chunks) rather than padded to whole 256-value blocks. Metal only; the loader sets it
 * on every qwen4 load. */
void ds4_gpu_qwen4_set_down_trimmed(bool on);
```

In `ds4_metal.m`, right after `qwen4_expert_row_bytes`:

```c
static bool g_qwen4_down_trimmed;

void ds4_gpu_qwen4_set_down_trimmed(bool on) {
    g_qwen4_down_trimmed = on;
}

/* Routed down rows: Q2_K and Q4_K pad ff_dim to whole 256-value blocks, except
 * trimmed Q4_K files, whose last block keeps its 16-byte header and the qs
 * chunks of the real values only.  Every kernel skips values from ff_dim on
 * before loading them, so only the row stride differs. */
static uint32_t qwen4_down_row_bytes(uint32_t type, uint32_t ff_dim) {
    if (type == 12u && g_qwen4_down_trimmed && (ff_dim % 256u) != 0) {
        return (ff_dim % 64u) ? 0u : ff_dim / 256u * 144u + 16u + (ff_dim % 256u) / 2u;
    }
    const uint32_t dim = (type == 10u || type == 12u) ? (ff_dim + 255u) / 256u * 256u : ff_dim;
    return qwen4_expert_row_bytes(type, dim);
}
```

At the five sites, replace the `weight_dim` or `down_dim` declaration and its `qwen4_expert_row_bytes` line with
one call:
- `ds4_gpu_qwen4_moe_down_tensor` and `ds4_gpu_qwen4_moe_mm_down_tensor`:
  `const uint32_t row_bytes = qwen4_down_row_bytes(weight_type, ff_dim);`
- `qwen4_stage_union`, `ds4_gpu_qwen4_stream_stage_layer_pipe` and `ds4_gpu_qwen4_moe_stream_layer`:
  `const uint32_t down_row_bytes = qwen4_down_row_bytes(down_type, ff_dim);`
- In `moe_stream_layer`, update the comment above the old lines so it names the helper: "down rows follow
  qwen4_down_row_bytes (matches ds4_gpu_qwen4_moe_down_tensor)".

Check: `grep -n "255u) / 256u \* 256u" ds4_metal.m` must no longer list lines 52225, 52409, 52813, 54280 or 54844.

- [ ] **Step 4: Run the kernel suite**

Run: `make tests/test_qwen4_kernels > /tmp/k.log 2>&1 && ./tests/test_qwen4_kernels > $WS/kernels.log 2>&1; echo exit $?; grep -h "trimmed" $WS/kernels.log; tail -1 $WS/kernels.log`
Here `$WS` is the plan workspace.
Expected:
- exit 0;
- six `trimmed Q4_K down F=… byte-exact` lines;
- the suite's final OK line.
GPU: run inside a GPU window, or as a short announced kernel-only window, because the suite allocates GPU
buffers.

- [ ] **Step 5: Commit**

```bash
git add ds4_gpu.h ds4_metal.m tests/test_qwen4_kernels.c
git commit -m "metal: qwen4 down row bytes through one helper; trimmed Q4_K down rows behind a per-load flag"
```

---

### Task 3: Loader support in ds4.c + C unit test

**Files:**
- Modify: `ds4.c`:
  - `tensor_nbytes` callers at the parse (about line 2870);
  - `routed_expert_row_bytes` (about line 5173);
  - `weights_validate_qwen4_layout` (about lines 5696-5803);
  - `qwen4_ref_row` Q4_K case (about line 70518).
- Create: `tests/test_qwen4_down_trim.c`
- Modify: `Makefile` (a rule like `tests/test_qwen4_prefill_pipe`, plus the target `test-qwen4-down-trim`)

**Interfaces:**
- Consumes: `ds4_gpu_qwen4_set_down_trimmed(bool)` from Task 2.
- Produces:
  - `static uint64_t q4k_row_bytes(uint64_t n)`, which returns 0 when n % 64 != 0;
  - `static bool tensor_nbytes_dims(uint32_t type, uint64_t dim0, uint64_t elements, uint64_t *bytes)`;
  - `static bool qwen4_down_is_trimmed(const ds4_tensor *t)`;
  - `static void qwen4_check_down_layouts(const bool *trimmed, const uint32_t *types, uint32_t n)`, which dies
    on a Q4_K mix.

- [ ] **Step 1: Write the failing C test** in `tests/test_qwen4_down_trim.c`

```c
/* Model-free checks for trimmed Q4_K down rows
 * (docs/superpowers/specs/2026-10-03-qwen4-q4k-down-trim-design.md). */
#include "../ds4.c"

static int failures;
#define CHECK(cond) do { \
        if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
    } while (0)

static void check_row_bytes(void) {
    CHECK(q4k_row_bytes(768) == 432);
    CHECK(q4k_row_bytes(640) == 368);
    CHECK(q4k_row_bytes(320) == 192);
    CHECK(q4k_row_bytes(704) == 400);
    CHECK(q4k_row_bytes(672) == 0);
    uint64_t b = 0;
    CHECK(tensor_nbytes_dims(DS4_TENSOR_Q4_K, 640, 640ull * 2560 * 512, &b) && b == 368ull * 2560 * 512);
    CHECK(tensor_nbytes_dims(DS4_TENSOR_Q4_K, 768, 768ull * 2560 * 512, &b) && b == 432ull * 2560 * 512);
    CHECK(!tensor_nbytes_dims(DS4_TENSOR_Q4_K, 672, 672ull * 4, &b));
    CHECK(tensor_nbytes_dims(DS4_TENSOR_Q8_0, 640, 640ull * 2560, &b) && b == 20ull * 34 * 2560);
    ds4_tensor t = { .ndim = 3, .dim = {640, 2560, 512}, .type = DS4_TENSOR_Q4_K };
    CHECK(routed_expert_row_bytes(&t) == 368);
    t.dim[0] = 768;
    CHECK(routed_expert_row_bytes(&t) == 432);
}

static void check_trimmed_predicate(void) {
    g_ds4_shape.n_ff_exp = 640;
    ds4_tensor t = { .ndim = 3, .dim = {640, 2560, 512}, .type = DS4_TENSOR_Q4_K };
    CHECK(qwen4_down_is_trimmed(&t));
    t.dim[0] = 768;
    CHECK(!qwen4_down_is_trimmed(&t));                 /* padded */
    t.dim[0] = 640; t.type = DS4_TENSOR_Q8_0;
    CHECK(!qwen4_down_is_trimmed(&t));                 /* MTP Q8_0 down: not a Q4_K row */
    g_ds4_shape.n_ff_exp = 512; t.type = DS4_TENSOR_Q4_K; t.dim[0] = 512;
    CHECK(!qwen4_down_is_trimmed(&t));                 /* whole blocks: nothing to trim */
}

/* the CPU reference reads a trimmed row exactly like the first n values of its padded twin */
static void check_ref_row(void) {
    enum { ROWS = 3 };
    uint8_t pad[ROWS * 432], trim[ROWS * 368];
    for (uint32_t i = 0; i < sizeof(pad); i++) pad[i] = (uint8_t)(i * 37u + 11u);
    for (uint32_t r = 0; r < ROWS; r++) {
        /* keep d and dmin finite and small: f16 0x2000 = 2^-7 in every block header */
        for (uint32_t b = 0; b < 3; b++) { pad[r * 432 + b * 144] = 0x00; pad[r * 432 + b * 144 + 1] = 0x20;
                                           pad[r * 432 + b * 144 + 2] = 0x00; pad[r * 432 + b * 144 + 3] = 0x20; }
        memcpy(trim + r * 368, pad + r * 432, 368);
    }
    ds4_model m = { .map = NULL };
    ds4_tensor tp = { .ndim = 2, .dim = {768, ROWS}, .type = DS4_TENSOR_Q4_K };
    ds4_tensor tt = { .ndim = 2, .dim = {640, ROWS}, .type = DS4_TENSOR_Q4_K };
    float a[768], b[640];
    for (uint32_t r = 0; r < ROWS; r++) {
        qwen4_ref_row_from(pad, &tp, r, a);
        qwen4_ref_row_from(trim, &tt, r, b);
        CHECK(memcmp(a, b, sizeof(b)) == 0);
    }
    (void)m;
}

int main(void) {
    check_row_bytes();
    check_trimmed_predicate();
    check_ref_row();
    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("test_qwen4_down_trim: ok\n");
    return 0;
}
```

`qwen4_ref_row` reads through `tensor_data(m, t)`. So that the test needs no fake model, Step 3 splits the Q4_K
case into `static void qwen4_ref_q4k_row(const uint8_t *data, uint64_t n, uint64_t row, float *out)`.
**Ruling:** the test calls that helper. Spell it in the test as
`qwen4_ref_q4k_row(pad, 768, r, a); qwen4_ref_q4k_row(trim, 640, r, b);` instead of the `qwen4_ref_row_from`
calls above, and drop the unused `m`, `tp` and `tt`.

Makefile, after the `test-qwen4-prefill-pipe` block:

```make
tests/test_qwen4_down_trim.o: tests/test_qwen4_down_trim.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_down_trim: tests/test_qwen4_down_trim.o $(filter-out ds4.o,$(CORE_OBJS))
ifeq ($(UNAME_S),Darwin)
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

.PHONY: test-qwen4-down-trim
test-qwen4-down-trim: tests/test_qwen4_down_trim
	./tests/test_qwen4_down_trim
```

Add `tests/test_qwen4_down_trim` to the `clean` rm list next to `tests/test_qwen4_prefill_pipe`.

- [ ] **Step 2: Run it**

Run: `make test-qwen4-down-trim 2>&1 | tail -5`
Expected: compile FAIL, with `q4k_row_bytes`, `tensor_nbytes_dims`, `qwen4_down_is_trimmed` and
`qwen4_ref_q4k_row` undeclared.

- [ ] **Step 3: Implement in ds4.c**

Put this after `tensor_nbytes` (about line 2630):

```c
/* Bytes of a Q4_K row of n values.  A row that ends inside a super-block
 * (trimmed qwen4 down experts, n % 256 != 0) keeps that block's 16-byte header
 * and only the 32-byte qs chunks of its real values (a chunk holds 64
 * values: sub-blocks 2k and 2k+1); 0 if n is not a multiple of 64. */
static uint64_t q4k_row_bytes(uint64_t n) {
    if (n % 64u) return 0;
    return n / 256u * 144u + ((n % 256u) ? 16u + (n % 256u) / 2u : 0u);
}

/* tensor_nbytes with the row length: Q4_K rows may end in a short block. */
static bool tensor_nbytes_dims(uint32_t type, uint64_t dim0, uint64_t elements, uint64_t *bytes) {
    if (type == DS4_TENSOR_Q4_K && dim0 != 0 && (dim0 % 256u) != 0) {
        const uint64_t row = q4k_row_bytes(dim0);
        if (row == 0 || elements % dim0 != 0 || elements / dim0 > UINT64_MAX / row) return false;
        *bytes = elements / dim0 * row;
        return true;
    }
    return tensor_nbytes(type, elements, bytes);
}
```

At the parse (about line 2870), replace `tensor_nbytes(t->type, t->elements, &t->bytes)` with
`tensor_nbytes_dims(t->type, t->ndim ? t->dim[0] : 0, t->elements, &t->bytes)`. Use the struct's actual field
name for the dim count, as read at that site.

In `routed_expert_row_bytes`, before `const gguf_type_info *info`:

```c
    if (t->type == DS4_TENSOR_Q4_K) {
        const uint64_t row = q4k_row_bytes(t->dim[0]);
        if (row == 0) ds4_die("routed expert Q4_K row is not a multiple of 64 values");
        return row;
    }
```

Before `weights_validate_qwen4_layout`:

```c
/* A routed down tensor stored as trimmed Q4_K rows: ff_dim values per row,
 * the last super-block cut after its header and real qs chunks. */
static bool qwen4_down_is_trimmed(const ds4_tensor *t) {
    return t && t->type == DS4_TENSOR_Q4_K && t->ndim == 3 && t->dim[0] == DS4_N_FF_EXP &&
           (DS4_N_FF_EXP % 256u) != 0 && (DS4_N_FF_EXP % 64u) == 0;
}

/* The Metal down stride is one per-model setting, so the Q4_K routed down
 * tensors must all be trimmed or all padded; other types do not count. */
static void qwen4_check_down_layouts(const bool *trimmed, const uint32_t *types, uint32_t n) {
    int first = -1;
    for (uint32_t i = 0; i < n; i++) {
        if (types[i] != DS4_TENSOR_Q4_K) continue;
        if (first < 0) first = trimmed[i] ? 1 : 0;
        else if ((trimmed[i] ? 1 : 0) != first) {
            fprintf(stderr, "ds4: routed Q4_K down experts mix trimmed and padded rows (layer slot %u)\n", i);
            exit(1);
        }
    }
}
```

Add a test for `qwen4_check_down_layouts` to the C test's `check_trimmed_predicate`. Only the accepting cases
can run in-process, because `exit(1)` kills the test, so test the accepting cases plus the predicate:

```c
    const bool tr[3] = { true, true, false };
    const uint32_t ty[3] = { DS4_TENSOR_Q4_K, DS4_TENSOR_Q4_K, DS4_TENSOR_Q8_0 };
    qwen4_check_down_layouts(tr, ty, 3);   /* Q8_0 MTP down beside trimmed Q4_K: accepted (returns) */
```

To cover the refusal, a forked child calls `qwen4_check_down_layouts` with `{true,false}` / `{Q4_K,Q4_K}`, and
the parent checks `WIFEXITED && WEXITSTATUS == 1`. Add `#include <sys/wait.h>` and:

```c
    pid_t pid = fork();
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        const bool mix[2] = { true, false };
        const uint32_t mt[2] = { DS4_TENSOR_Q4_K, DS4_TENSOR_Q4_K };
        qwen4_check_down_layouts(mix, mt, 2);
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 1);
```

In `weights_validate_qwen4_layout`:
- Declare `bool down_trimmed[DS4_MAX_LAYER] = {0}; uint32_t down_types[DS4_MAX_LAYER] = {0}; uint32_t n_down = 0;`
  before the layer loop.
- Replace the `down_width` block with:

```c
        /* Q2_K and Q4_K down rows store 640 logical inputs in three 256-value
         * blocks unless the Q4_K rows are trimmed (spec 2026-10-03). Activations
         * remain 640 wide; only the weight row stride differs. */
        const bool trimmed = qwen4_down_is_trimmed(l->ffn_down_exps);
        const uint32_t down_width = !trimmed && (l->ffn_down_exps->type == DS4_TENSOR_Q2_K ||
                                                 l->ffn_down_exps->type == DS4_TENSOR_Q4_K) ?
            (DS4_N_FF_EXP + 255u) / 256u * 256u : DS4_N_FF_EXP;
        tensor_expect_qwen4_expert_layout(l->ffn_down_exps, down_width, DS4_N_EMBD, DS4_N_EXPERT);
        down_trimmed[n_down] = trimmed;
        down_types[n_down++] = l->ffn_down_exps->type;
```

After the layer loop:

```c
    qwen4_check_down_layouts(down_trimmed, down_types, n_down);
    bool any_trimmed = false;
    for (uint32_t i = 0; i < n_down; i++) any_trimmed |= down_trimmed[i];
#ifdef DS4_HAS_QWEN4_METAL
    ds4_gpu_qwen4_set_down_trimmed(any_trimmed);
#elif defined(DS4_HAS_QWEN4_GPU)
    if (any_trimmed) {
        fprintf(stderr, "ds4: trimmed Q4_K down rows need the Metal backend\n");
        exit(1);
    }
#endif
```

**Ruling (planned):** if the loop's layer bounds are `layer_start..layer_end`, size the arrays by
`DS4_MAX_LAYER`, which is already a compile-time cap used in this file. If `DS4_MAX_LAYER` is not visible there,
use the same bound the layer arrays use.

Split the Q4_K case of `qwen4_ref_row` into a helper placed above `qwen4_ref_row`:

```c
static void qwen4_ref_q4k_row(const uint8_t *data, uint64_t n, uint64_t row, float *out) {
    const uint64_t row_bytes = q4k_row_bytes(n);
    if (row_bytes == 0) ds4_die("qwen4 reference: Q4_K row is not a multiple of 64 values");
    const uint8_t *p = data + row * row_bytes;
    for (uint64_t b = 0; b * 256u < n; b++, p += 144u) {
        uint16_t dh, mh;
        memcpy(&dh, p, 2);
        memcpy(&mh, p + 2, 2);
        const float d = f16_to_f32(dh), dmin = f16_to_f32(mh);
        const uint8_t *sc = p + 4;
        for (uint32_t g = 0; g < 8u && b * 256u + g * 32u < n; g++) {
            uint32_t s, mn;
            if (g < 4u) { s = sc[g] & 63u; mn = sc[g + 4] & 63u; }
            else { s = (sc[g + 4] & 0xFu) | ((sc[g - 4] & 0xC0u) >> 2); mn = (sc[g + 4] >> 4) | ((sc[g] & 0xC0u) >> 2); }
            const float ds = d * (float)s, dm = dmin * (float)mn;
            const uint8_t *qs = p + 16 + (g >> 1) * 32u;
            const uint32_t shift = (g & 1u) * 4u;
            for (uint32_t j = 0; j < 32u; j++) out[b * 256u + g * 32u + j] = ds * (float)((qs[j] >> shift) & 0xFu) - dm;
        }
    }
}
```

The case body becomes `qwen4_ref_q4k_row((const uint8_t *)tensor_data(m, t), n, row, out); break;`. Keep whatever
the old case did after its block loop, if anything.

- [ ] **Step 4: Run it**

Run: `make test-qwen4-down-trim 2>&1 | tail -3`
Expected: `test_qwen4_down_trim: ok`.

- [ ] **Step 5: Build everything and run the CPU-side suites**

Run: `make -j8 ds4 ds4-bench ds4-server > $WS/build.log 2>&1; echo $?; make test-qwen4-prefill-pipe 2>&1 | tail -2`
Expected: 0, and the prefill-pipe test OK.

- [ ] **Step 6: Commit**

```bash
git add ds4.c Makefile tests/test_qwen4_down_trim.c
git commit -m "ds4: load trimmed Q4_K routed down rows (parse size, stream bytes, validator, CPU reference)"
```

---

### Task 4: Build the trimmed GGUF (disk only, no GPU)

**Files:** none in git. Output: the trimmed file plus `.json`, and a log at
`/Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/convert.log`.

- [ ] **Step 1: Check free disk**

Run: `df -g /Users/dongnh/orca/workspaces/ds4-metal-data | tail -1`
Expected: at least 60 GiB free.

- [ ] **Step 2: Convert**

```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/kv-grow
/usr/bin/time -l python3 tools/qwen4_trim_down_pad.py \
  /Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf \
  /Users/dongnh/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownTrim-DenseQ4Kselimat-MTP.gguf \
  > /Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/convert.log 2>&1
```

Expected: `wrote …: N tensors, 48 down tensors trimmed`, with the output 3.75 GiB smaller than the input.
`.json` lists 48 `trimmed_from: [768, 2560, 512]` entries.

Ruling: run it when no peer holds a production window. The read of the 52 GB PROD file evicts page cache, so
announce it to the peers like a window.

- [ ] **Step 3: Check the sizes**

```bash
python3 - <<'EOF'
import json
m = json.load(open("/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownTrim-DenseQ4Kselimat-MTP.gguf.json"))
tr = [k for k, v in m["tensors"].items() if v["trimmed_from"]]
print(len(tr), sorted(set(m["tensors"][k]["bytes"] for k in tr)))
EOF
```

Expected: `48 [482344960]`.

---

### Task 5: End-to-end identity (GPU window 1)

**Files:** none in git. Script: `/Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/run-identity.sh`,
modelled on `merge-gate/run-gate-f51c.sh`.

- [ ] **Step 1: Write the script**

```zsh
#!/bin/zsh
# Trimmed Q4_K down identity: base (develop ef4d897c) x PROD, new x PROD, new x trimmed.
# greedy 3 prompts (decode + MTP, streamed layers 32-47) and perplexity on 3 texts (prefill GEMM).
# Also the stream-cache expert count from the startup log. Own gateway hold. Never kill -9 a Metal process.
G=/Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim
S=/Users/dongnh/orca/workspaces/ds4-metal-data/sp3
P=/Users/dongnh/orca/workspaces/ds4-metal-data/sp3gemm/ppl
R=/Users/dongnh/orca/workspaces/ds4-metal/kv-grow
O=$G/identity
M=/Users/dongnh/.local/share/ai-gateway/ds4-models
PLE=$M/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
PROD=$M/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf
TRIM=/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownTrim-DenseQ4Kselimat-MTP.gguf
PROD_ENV=(DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0 DS4_QWEN4_KV_GROW=1)
PROD_FLAGS=(--metal --ple $PLE --prefill-chunk 2048 --ssd-streaming --ssd-streaming-cache-experts 6GB)
PROMPTS=("Write a Python function that checks whether a number is prime."
         "Giải thích ngắn gọn vì sao bầu trời có màu xanh."
         "List three differences between TCP and UDP.")
stamp() { date '+%H:%M:%S'; }
for f in $PLE $PROD $TRIM $G/base/ds4 $R/ds4 $P/en.txt $P/code.txt $P/vi-new.txt; do
  [ -e $f ] || { echo "== missing $f"; exit 1; }
done
mkdir -p $O
echo "== start $(stamp)"
DEADLINE=${DEADLINE:?set DEADLINE=HHMM} zsh $S/hold-gateway.sh > $G/hold-identity.out 2>&1 &
hold=$!
for i in {1..120}; do
  grep -q "paused, GPU free" $G/hold-identity.out && break
  grep -q "ABORT" $G/hold-identity.out && { echo "== hold aborted $(stamp)"; wait $hold; exit 1; }
  sleep 10
done
grep -q "paused, GPU free" $G/hold-identity.out || { echo "== hold not paused $(stamp)"; touch $S/release-gateway; wait $hold; exit 1; }
echo "== gateway paused $(stamp)"
arm() {   # tag binary model
  for i in 1 2 3; do
    env $PROD_ENV $2 -m $3 $PROD_FLAGS -c 262144 --nothink --temp 0 -n 128 -p "${PROMPTS[$i]}" \
      > $O/$1-g$i.txt 2> $O/$1-g$i.log
  done
  for t in en code vi-new; do
    env $PROD_ENV $2 -m $3 $PROD_FLAGS -c 4096 -n 2048 --perplexity-file $P/$t.txt > $O/$1-ppl-$t.out 2> $O/$1-ppl-$t.log
  done
  echo "$1 done $(stamp): $(grep -ho 'avg_nll=[0-9.]*' $O/$1-ppl-*.out | tr '\n' ' ')"
}
cd $R
echo "== kernels $(stamp)"; ./tests/test_qwen4_kernels > $O/kernels.log 2>&1; echo "kernels exit $?"; grep -h "trimmed" $O/kernels.log
arm base-prod $G/base/ds4 $PROD
arm new-prod ./ds4 $PROD
arm new-trim ./ds4 $TRIM
touch $S/release-gateway
wait $hold
tail -3 $G/hold-identity.out
echo "== compare"
for i in 1 2 3; do
  cmp -s $O/base-prod-g$i.txt $O/new-prod-g$i.txt && echo "g$i new-prod == base" || echo "g$i new-prod DIFFERS"
  cmp -s $O/base-prod-g$i.txt $O/new-trim-g$i.txt && echo "g$i new-trim == base" || echo "g$i new-trim DIFFERS"
done
for t in en code vi-new; do
  for a in new-prod new-trim; do
    diff -q <(grep -o 'avg_nll=[0-9.]*' $O/base-prod-ppl-$t.out) <(grep -o 'avg_nll=[0-9.]*' $O/$a-ppl-$t.out) >/dev/null \
      && echo "ppl $t $a == base" || echo "ppl $t $a DIFFERS"
  done
done
grep -h -i "expert cache\|slots\|planned" $O/new-prod-g1.log $O/new-trim-g1.log | head -12
echo "== done $(stamp)"
```

- [ ] **Step 2: Announce the window to the peers.** Wait until no production window is active (CodeTab's ends
  with their "done"). Then run with `DEADLINE` set to now + 60 min:

`DEADLINE=HHMM zsh /Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/run-identity.sh > /Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/run-identity.log 2>&1`
(background; monitor the log)

Expected:
- kernels exit 0, with six trimmed lines;
- every `g$i … == base`;
- every `ppl … == base`;
- the trimmed log reports more experts in the 6 GB cache.
Any DIFFERS line is a finding: stop and use systematic-debugging. Do not change the comparison.

- [ ] **Step 3: Send "done" plus backend_ok to the peers.** Record the results in
  `/Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/NOTES.md`.

---

### Task 6: Warm-controlled speed A/B (GPU window 2, may share window 1 if time allows)

**Files:** none in git. Script: `/Users/dongnh/orca/workspaces/ds4-metal-data/q4k-trim/run-speed.sh`.

- [ ] **Step 1: Write the script.**
  - Same header and hold as Task 5.
  - `BENCH=(--prompt-file $R/tests/long_context_story_prompt.txt --ctx-start 8192 --ctx-max 8192 --gen-tokens 128)`.
  - Arms, in order: prod-a1, prod-a2, trim-a1, trim-a2, then trim-b1, trim-b2, prod-b1, prod-b2.
  - Every arm uses `./ds4-bench` (new binary): `env $PROD_ENV ./ds4-bench -m $MODEL $PROD_FLAGS $BENCH --csv $O/w-$tag.csv`.
  - Then a cold-start pair per model: `cat` the PLE and the other model to evict, then a short prompt with
    `./ds4 … -n 16`, timed with `EPOCHREALTIME` exactly as `prefill-spike/run-warm.sh` does.

- [ ] **Step 2: Run it in an announced window.** Then compare the warm second runs: prod-a2 vs trim-a2, and
  trim-b2 vs prod-b2. Report the prefill t/s, decode t/s and cold-start wall time deltas.

Expected: a measurement, with no required sign. A loss larger than noise (more than 3% in both orders) is a
finding to explain in the results doc. It does not count as a failure of the feature.

- [ ] **Step 3: Send "done" plus backend_ok.** Add the numbers to NOTES.md.

---

### Task 7: Qwen gate on the trimmed file (GPU window 3)

**Files:**
- Modify: `speed-bench/qwen-regression/qwen_gate.py` (`check` gets `--branch-model`)
- Modify: `speed-bench/qwen-regression/run.sh` (passes `BRANCH_MODEL` through)
- Test: `speed-bench/qwen-regression/test_qwen_gate.py` (new, model-free)

**Interfaces:**
- Produces: `qwen_gate.retarget_model(cmd: list, path: str | None) -> list`, which returns cmd with the value
  after `-m` replaced, or cmd unchanged when path is None.

- [ ] **Step 1: Write the failing test**

```python
import unittest

import qwen_gate as q


class RetargetTest(unittest.TestCase):
    def test_replaces_model(self):
        self.assertEqual(q.retarget_model(["ds4-server", "-m", "a.gguf", "-c", "8"], "b.gguf"),
                         ["ds4-server", "-m", "b.gguf", "-c", "8"])

    def test_none_keeps(self):
        cmd = ["ds4-server", "-m", "a.gguf"]
        self.assertEqual(q.retarget_model(cmd, None), cmd)

    def test_missing_flag_refused(self):
        with self.assertRaises(SystemExit):
            q.retarget_model(["ds4-server"], "b.gguf")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it**

Run: `cd speed-bench/qwen-regression && python3 -m unittest test_qwen_gate -v`
Expected: FAIL, `AttributeError: … retarget_model`.

- [ ] **Step 3: Implement**

```python
def retarget_model(cmd, path):
    """The branch side of a trimmed-model gate runs the same command on another GGUF."""
    if path is None:
        return cmd
    if "-m" not in cmd[:-1]:
        raise SystemExit("qwen_gate: registry command has no -m to retarget")
    out = list(cmd)
    out[out.index("-m") + 1] = path
    return out
```

- Add `chk.add_argument("--branch-model", default=None, help="GGUF for the branch servers (default: the registry model)")`.
- Every place `check` builds a branch-side command from `registry_command(REGISTRY, bin_dir, ...)` wraps it as
  `retarget_model(cmd, args.branch_model)`. These are the fast replies and the paired speed runs' branch servers.
  The PROD side and `record` stay on the registry model.
- If the needle (`long`) step runs the branch binary, retarget it the same way.
- In `run.sh`, append `${BRANCH_MODEL:+--branch-model "$BRANCH_MODEL"}` to the `qwen_gate.py check` line.

- [ ] **Step 4: Run the test**

Run: `cd speed-bench/qwen-regression && python3 -m unittest test_qwen_gate -v`
Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add speed-bench/qwen-regression/qwen_gate.py speed-bench/qwen-regression/run.sh speed-bench/qwen-regression/test_qwen_gate.py
git commit -m "qwen-regression: --branch-model runs the branch servers on another GGUF (trimmed down gate)"
```

- [ ] **Step 6: Run the full gate in an announced window.** The gate stops ds4 processes itself, but it needs
  the gateway paused, so wrap it in `hold-gateway.sh` like Task 5's script:

`BRANCH_MODEL=<trimmed path> speed-bench/qwen-regression/run.sh full > $G/gate-full.log 2>&1`
Expected:
- the fast tier byte-identical to the baseline;
- decode at least 97% of PROD;
- wired at most baseline + 0.5 GiB (trimmed should be lower);
- needle found.
A paired-speed miss alone is rerun once, per the README.

- [ ] **Step 7: Send "done" plus backend_ok.**

---

### Task 8: Results doc, memory, review, finish

**Files:**
- Create: `speed-bench/nextgen-eval/results/2026-10-03-q4k-down-trim.md`
- Modify: memory `nextgen-qwen-design.md` or a new memory `q4k-down-trim.md` plus a MEMORY.md line

- [ ] **Step 1: Write the results doc.** Include:
  - the format;
  - the sizes measured on the real file;
  - the identity table (Task 5);
  - the warm A/B (Task 6), cold-start pair included;
  - the gate result (Task 7);
  - the freed memory (resident and cache);
  - the deploy steps that wait for the user: HF upload to dongnhdev, `prod/q4k-down-trim-YYYYMMDD` from develop,
    and the registry `-m` path.
- [ ] **Step 2: Commit the doc.**
- [ ] **Step 3: Final whole-branch review** (executing-plans' final review on the most capable model). Then run
  the fix pass per the skill.
- [ ] **Step 4: Run finishing-a-development-branch.** Present the options and wait for the user's choice. Do not
  merge or push on autonomy.
