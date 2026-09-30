# GSQ-RCO Tensor Types and ISTA Repack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ds4 runs ISTA-DASLab's GSQ-RCO IQ3_XXS Qwen3.8-Flash-Next, repacked with Ivan's MTP head, on Metal, correct first.

**Architecture:** Nine quant types (Q5_K, Q6_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0) get CPU row dequantizers (`ds4_quants.h`, the reference) and lane helpers inside the one generic Metal row dot every qwen4 kernel already calls (`qwen4_row_dot`). Dense tensors of those types route through the existing multi-GEMV kernel; routed experts through the existing generic MoE kernels. A Python repack tool builds one ds4 GGUF from ISTA's shard 1 plus Ivan's `blk.48`.

**Tech Stack:** C99 (ds4.c), Metal Shading Language (metal/*.metal), Objective-C (ds4_metal.m), Python 3.9 + numpy (tools), unittest.

**Spec:** `docs/superpowers/specs/2026-09-30-gsq-rco-types-design.md`

## Global Constraints

- Mac only. Git holds source and small fixtures; GGUFs live in `~/orca/workspaces/ds4-metal-data` and on HF.
- No GPU work without the user's go-ahead; pause the gateway stack for model runs and restore it.
- Never `kill -9` a Metal process. Code, docs and commits in English.
- Upstream source for every port: ggml-org/llama.cpp at `931351ea50dfdd3ee249606f655eef2e9a629daf` (MIT); keep the notice in ported files.
- Work in the worktree `~/orca/workspaces/ds4-metal/sp3`, branch `feature/nextgen-sp3` from `feature/nextgen-qwen`. The kv-grow worktree's binaries and `metal/` serve the sub-project 4 run and must not change while it runs.
- While the sub-project 4 GPU batch runs (started 2026-09-30 12:04, about 8 h): no GPU, no multi-GB disk IO (the repack run waits); builds and CPU tests are fine.
- Every commit ends with the two trailer lines `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn`.

## Spec reconciliations (found reading the code; binding for this plan)

1. The qwen4 graph never calls ggml's `kernel_mul_mv_id`/`kernel_mul_mm_id` templates: its MoE and dense kernels (`kernel_qwen4_moe_mid`, `kernel_qwen4_moe_down`, the IQ2 NR kernels' shared slot, `kernel_qwen4_multi_gemv`, the GDN front) all call `qwen4_row_dot(row, x, weight_type, in_dim, tiisg)` in `metal/qwen4.metal`. The port therefore lands as per-type lane helpers inside `qwen4_row_dot`; no new kernels and no new pipelines (spec Engine items 3-5).
2. `token_embd` is gathered on the CPU by `qwen4_ref_row` (ds4.c), so IQ3_S needs a CPU row, not a Metal `get_rows`.
3. The tiled prefill GEMM (`qwen4_expert_type_has_mm`) does not take the new types: prefill of those layers runs the per-token row kernels. Correct, slower; Task 9 measures it.
4. The SSD streaming expert cache serves only IQ2_XXS gate/up with Q2_K/Q4_K down (`qwen4_stream_expert_cache_addr_layout_supported`); ISTA layers keep mapped views. `routed_expert_block_bytes` learns the new sizes so no sizing path dies (spec Engine item 7).
5. ds4 does not read `ds4.qwen4.down.*`; the loader picks the down width from the type (padding only for Q2_K/Q4_K). The repack still writes the keys (informational).
6. The GPU graph requires `hc_{attn,ffn}_{up,inject}` in F16/F32/Q8_0 and the loader requires `ffn_gate_inp` and `ffn_gate_inp_shexp` in F32; ISTA stores all six as BF16. The repack converts the four hc tensors BF16→F16 when every value round-trips exactly (else F32) and the two router tensors BF16→F32 (always exact).
7. The repack uses numpy (installed, 2.0.2) for those conversions; pure-Python loops over 300M values are too slow.
8. Fixture comparisons allow 2 ulp: ds4 builds with `-ffast-math` (FMA contraction), and ggml's own C and Python dequantizers differ at that level.
9. `DS4_QWEN4_MTP_DRAFT_VOCAB` needs a Q8_0 output head; ISTA's is Q5_K, so the draft-vocab speedup is off for this model (noted for sub-project 5).
10. The decode MoE kernels read the shared expert's gate and up with one type (`ffn_gate_shexp->type`); ISTA's shared gate and up differ in 31 of 48 layers. Those layers run the shared expert through the dense path at every row count (Task 8). Found while planning; the spec did not list it.
11. The model-level correctness check is ds4's own `--first-token-test` with `DS4_QWEN4_GPU=1`: the Metal graph against the CPU reference forward (`qwen4_ref_forward_token`) on the same tokens, logits compared.

## Review Focus

1. Layers whose routed experts are IQ2_XXS but whose shared expert is a new type (9 ISTA layers): the M5 IQ2 NR mid kernel computes the shared slot with `qwen4_row_dot` specialized on the shared type — it must be right. Pinned by Task 7's MoE case `(16, 20, 23, 20)`.
2. 640-wide down rows of IQ4_NL and Q2_0 must be read unpadded (row stride 360 / 180 bytes, not a 768-wide stride). Pinned by Task 7's MoE cases.
3. Non-Metal builds (`make cpu`, CUDA) must compile and refuse the new types in the GPU graph with a message, not crash. Pinned by Task 8's `make cpu` step and `qwen4_graph_gsq_ok`.
4. A BF16 hc tensor with a value F16 cannot hold exactly (subnormal or > 65504) must become F32, not a rounded F16. Pinned by Task 5's `test_hc_falls_back_to_f32`.
5. A truncated or wrong source file must make the repack refuse, never write a short tensor. Pinned by Task 5's `test_truncated_source_refused`.

---

### Task 1: Quant type table

**Files:**
- Modify: `ds4.c` (enum at the `DS4_TENSOR_*` block ~l.2451; `gguf_types` ~l.2418; `tensor_is_routed_expert_type` ~l.5086; `routed_expert_block_bytes` ~l.5096; new `ds4_gguf_type_block` after `tensor_type` ~l.2574)
- Modify: `ds4.h` (declaration next to `ds4_qwen4_yarn_factor`)
- Test: `tests/ds4_test.c` (`test_quant_types`, entry `--quant-types`)

**Interfaces:**
- Produces: `int ds4_gguf_type_block(uint32_t type, uint32_t *block_elems, uint32_t *block_bytes);` (0 = known type); enum values `DS4_TENSOR_IQ2_XS=17, IQ3_XXS=18, IQ4_NL=20, IQ3_S=21, IQ2_S=22, IQ4_XS=23, Q2_0=42`.

- [ ] **Step 1: Write the failing test** (in `tests/ds4_test.c`, above `test_qwen_yarn_policy`)

```c
static void test_quant_types(void) {
    /* ggml-common.h @931351ea block sizes of every quant type ds4 sizes tensors with */
    static const struct { uint32_t type, elems, bytes; } want[] = {
        {2, 32, 18}, {3, 32, 20}, {8, 32, 34}, {10, 256, 84}, {12, 256, 144}, {13, 256, 176},
        {14, 256, 210}, {16, 256, 66}, {17, 256, 74}, {18, 256, 98}, {19, 256, 50}, {20, 32, 18},
        {21, 256, 110}, {22, 256, 82}, {23, 256, 136}, {29, 256, 56}, {39, 32, 17}, {42, 64, 18},
    };
    for (size_t i = 0; i < sizeof(want) / sizeof(want[0]); i++) {
        uint32_t e = 0, b = 0;
        TEST_ASSERT(ds4_gguf_type_block(want[i].type, &e, &b) == 0);
        if (e != want[i].elems || b != want[i].bytes) {
            fprintf(stderr, "ds4-test: type %u is %u/%u, ggml has %u/%u\n",
                    want[i].type, e, b, want[i].elems, want[i].bytes);
        }
        TEST_ASSERT(e == want[i].elems && b == want[i].bytes);
    }
    uint32_t e = 0, b = 0;
    TEST_ASSERT(ds4_gguf_type_block(41, &e, &b) != 0);    /* no such type */
    TEST_ASSERT(ds4_gguf_type_block(1000, &e, &b) != 0);
}
```

and the entry after `--dir-steering-rows` in `test_entries[]`:

```c
    {"--quant-types", "quant-types", "GGUF quant block sizes match ggml (no model)", test_quant_types},
```

and the declaration in `ds4.h` after `ds4_qwen4_yarn_env_factor`:

```c
/* Block geometry of a GGUF tensor type (weights per block, bytes per block);
 * returns 0, or -1 for a type ds4 does not know. */
int ds4_gguf_type_block(uint32_t type, uint32_t *block_elems, uint32_t *block_bytes);
```

- [ ] **Step 2: Run it to verify it fails**

Run: `make ds4_test 2>&1 | tail -3`
Expected: FAIL to link with `Undefined symbols ... _ds4_gguf_type_block`.

- [ ] **Step 3: Implement**

In `ds4.c`, the enum gains (keep numeric order):

```c
    DS4_TENSOR_IQ2_XS   = 17,
    DS4_TENSOR_IQ3_XXS  = 18,
    DS4_TENSOR_IQ4_NL   = 20,
    DS4_TENSOR_IQ3_S    = 21,
    DS4_TENSOR_IQ2_S    = 22,
    DS4_TENSOR_IQ4_XS   = 23,
```
after `DS4_TENSOR_IQ2_XXS = 16,` and `DS4_TENSOR_Q2_0 = 42,` after `DS4_TENSOR_MXFP4 = 39,`.

In `gguf_types`, replace the two wrong entries and add Q2_0:

```c
    [19] = {"iq1_s",  256,  50},
    [20] = {"iq4_nl",  32,  18},
```
```c
    [39] = {"mxfp4",   32,  17},
    [42] = {"q2_0",    64,  18},
```

After `tensor_type()`:

```c
int ds4_gguf_type_block(uint32_t type, uint32_t *block_elems, uint32_t *block_bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return -1;
    if (block_elems) *block_elems = info->block_elems;
    if (block_bytes) *block_bytes = info->block_bytes;
    return 0;
}
```

`tensor_is_routed_expert_type` gains `|| type == DS4_TENSOR_IQ2_XS || type == DS4_TENSOR_IQ2_S || type == DS4_TENSOR_IQ3_XXS || type == DS4_TENSOR_IQ3_S || type == DS4_TENSOR_IQ4_NL || type == DS4_TENSOR_IQ4_XS || type == DS4_TENSOR_Q2_0`, and `routed_expert_block_bytes` gains:

```c
    case DS4_TENSOR_IQ2_XS:  return 74;
    case DS4_TENSOR_IQ2_S:   return 82;
    case DS4_TENSOR_IQ3_XXS: return 98;
    case DS4_TENSOR_IQ3_S:   return 110;
    case DS4_TENSOR_IQ4_NL:  return 18;
    case DS4_TENSOR_IQ4_XS:  return 136;
    case DS4_TENSOR_Q2_0:    return 18;
```

- [ ] **Step 4: Run it to verify it passes**

Run: `make ds4_test >/dev/null && ./ds4_test --quant-types`
Expected: `quant-types: OK`, `ds4 tests: ok`.

- [ ] **Step 5: Commit**

```bash
git add ds4.c ds4.h tests/ds4_test.c
git commit -m "ds4: fix the iq4_nl/iq1_s block sizes, add q2_0 and the IQ tensor types"
```

---

### Task 2: IQ grid tables

**Files:**
- Create: `tools/gen_iq_tables.py`, generated `ds4_iq_tables.h`, generated `metal/iq_tables.metal`
- Modify: `ds4.c` (include after the IQ2_XXS tables ~l.1320), `ds4_metal.m` (source list ~l.5031), `Makefile` (l.492 and l.558 prerequisites)
- Test: `tests/ds4_test.c` (`test_quant_types` spot checks)

**Interfaces:**
- Consumes: nothing.
- Produces (C, `ds4_iq_tables.h`): `static const uint64_t iq2xs_grid[512], iq2s_grid[1024]; static const uint32_t iq3xxs_grid[256], iq3s_grid[512]; static const int8_t kvalues_iq4nl[16];`
- Produces (Metal, `metal/iq_tables.metal`): `ds4_metal_iq2xs_grid` (ulong[512]), `ds4_metal_iq2s_grid` (ulong[1024]), `ds4_metal_iq3xxs_grid` (uint[256]), `ds4_metal_iq3s_grid` (uint[512]), `ds4_metal_kvalues_iq4nl` (char[16]).

- [ ] **Step 1: Write the failing test** — `tests/ds4_test.c` cannot see ds4.c statics, so the tables are read through an accessor. Add to `ds4.h`:

```c
/* Test hook: entry i of a GSQ-RCO grid table ("iq2xs", "iq2s", "iq3xxs", "iq3s", "iq4nl"), or 0. */
uint64_t ds4_iq_table_entry(const char *table, uint32_t i);
```

and append to `test_quant_types` (first and last entries, read from ggml-common.h @931351ea):

```c
    TEST_ASSERT(ds4_iq_table_entry("iq2xs", 0) == 0x0808080808080808ULL);
    TEST_ASSERT(ds4_iq_table_entry("iq2xs", 511) == 0x2b2b2b2b2b2b2b2bULL);
    TEST_ASSERT(ds4_iq_table_entry("iq2s", 0) == 0x0808080808080808ULL);
    TEST_ASSERT(ds4_iq_table_entry("iq2s", 1023) == 0x2b2b2b2b2b2b2b2bULL);
    TEST_ASSERT(ds4_iq_table_entry("iq3xxs", 0) == 0x04040404u);
    TEST_ASSERT(ds4_iq_table_entry("iq3xxs", 255) == 0x3e341c04u);
    TEST_ASSERT(ds4_iq_table_entry("iq3s", 0) == 0x01010101u);
    TEST_ASSERT(ds4_iq_table_entry("iq3s", 511) == 0x0f0f0101u);
    TEST_ASSERT(ds4_iq_table_entry("iq4nl", 0) == (uint64_t)(int64_t)-127);
    TEST_ASSERT(ds4_iq_table_entry("iq4nl", 15) == 113u);
    TEST_ASSERT(ds4_iq_table_entry("iq2xs", 512) == 0);   /* out of range */
```

- [ ] **Step 2: Run it to verify it fails**

Run: `make ds4_test 2>&1 | tail -3`
Expected: link FAIL, `Undefined symbols ... _ds4_iq_table_entry`.

- [ ] **Step 3: Write the generator** `tools/gen_iq_tables.py`

```python
"""Emit ds4's GSQ-RCO grid tables from ggml's ggml-common.h (MIT).

usage: python3 tools/gen_iq_tables.py PATH/TO/ggml-common.h
Writes ds4_iq_tables.h (C) and metal/iq_tables.metal (Metal constant memory) at the repo root."""
import pathlib
import re
import sys

UPSTREAM = "ggml-org/llama.cpp@931351ea50dfdd3ee249606f655eef2e9a629daf ggml/src/ggml-common.h"
NOTICE = ("Generated by tools/gen_iq_tables.py from %s. Do not edit.\n"
          " * Copyright (c) 2023-2026 The ggml authors. MIT License." % UPSTREAM)
# (C type, Metal type, name, entries, values per line)
TABLES = [("uint64_t", "ulong", "iq2xs_grid", 512, 4), ("uint64_t", "ulong", "iq2s_grid", 1024, 4),
          ("uint32_t", "uint", "iq3xxs_grid", 256, 8), ("uint32_t", "uint", "iq3s_grid", 512, 8),
          ("int8_t", "char", "kvalues_iq4nl", 16, 16)]
ROOT = pathlib.Path(__file__).resolve().parent.parent


def table_values(src, ctype, name, n):
    m = re.search(r"GGML_TABLE_BEGIN\(%s,\s*%s,\s*%d\)(.*?)GGML_TABLE_END\(\)" % (ctype, name, n), src, re.S)
    if not m:
        raise SystemExit("table %s not found" % name)
    vals = re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", m.group(1))
    if len(vals) != n:
        raise SystemExit("table %s: %d values, want %d" % (name, len(vals), n))
    return [v + ("ULL" if ctype == "uint64_t" else "u" if ctype == "uint32_t" else "") for v in vals]


def body(vals, per_line):
    return "\n".join("    " + ", ".join(vals[i:i + per_line]) + "," for i in range(0, len(vals), per_line))


def main(argv):
    src = pathlib.Path(argv[1]).read_text()
    c = ["/* %s */\n#pragma once\n#include <stdint.h>\n" % NOTICE]
    metal = ["/* %s */\n" % NOTICE]
    for ctype, mtype, name, n, per in TABLES:
        vals = table_values(src, ctype, name, n)
        c.append("static const %s %s[%d] = {\n%s\n};\n" % (ctype, name, n, body(vals, per)))
        mvals = [v[:-3] if v.endswith("ULL") else v[:-1] if v.endswith("u") else v for v in vals]
        metal.append("static constant %s ds4_metal_%s[%d] = {\n%s\n};\n" % (mtype, name, n, body(mvals, per)))
    (ROOT / "ds4_iq_tables.h").write_text("\n".join(c))
    (ROOT / "metal" / "iq_tables.metal").write_text("\n".join(metal))


if __name__ == "__main__":
    main(sys.argv)
```

Run it on the pinned upstream header:

```bash
curl -sL -o "$SCRATCH/ggml-common.h" https://raw.githubusercontent.com/ggml-org/llama.cpp/931351ea50dfdd3ee249606f655eef2e9a629daf/ggml/src/ggml-common.h
python3 tools/gen_iq_tables.py "$SCRATCH/ggml-common.h"
```
(`$SCRATCH` = the session scratchpad.) Expected: `ds4_iq_tables.h` and `metal/iq_tables.metal` exist; `grep -c 0x ds4_iq_tables.h` > 2000.

- [ ] **Step 4: Wire the tables**

`ds4.c`, right after the `iq2xxs_signed_grid_init` block (~l.1330): `#include "ds4_iq_tables.h"`. After `ds4_gguf_type_block`:

```c
uint64_t ds4_iq_table_entry(const char *table, uint32_t i) {
    if (!strcmp(table, "iq2xs") && i < 512) return iq2xs_grid[i];
    if (!strcmp(table, "iq2s") && i < 1024) return iq2s_grid[i];
    if (!strcmp(table, "iq3xxs") && i < 256) return iq3xxs_grid[i];
    if (!strcmp(table, "iq3s") && i < 512) return iq3s_grid[i];
    if (!strcmp(table, "iq4nl") && i < 16) return (uint64_t)(int64_t)kvalues_iq4nl[i];
    return 0;
}
```

`ds4_metal.m` `required_sources`: insert before the `DS4_METAL_QWEN4_SOURCE` line

```objc
        @[@"DS4_METAL_IQ_TABLES_SOURCE",  @"metal/iq_tables.metal"],
```

`Makefile`: append ` ds4_iq_tables.h ds4_quants.h` to the prerequisites of `ds4.o:` (l.492), `ds4_cpu.o:` (l.558) and `ds4_cpu_test_hooks.o:` (l.916). (`ds4_quants.h` arrives in Task 4; create it now as an empty file with the notice comment so the rules resolve.)

- [ ] **Step 5: Run the test**

Run: `make ds4_test >/dev/null && ./ds4_test --quant-types`
Expected: `quant-types: OK`.

- [ ] **Step 6: Commit**

```bash
git add tools/gen_iq_tables.py ds4_iq_tables.h ds4_quants.h metal/iq_tables.metal ds4.c ds4.h ds4_metal.m Makefile tests/ds4_test.c
git commit -m "ds4: generated GSQ-RCO grid tables for the CPU and Metal"
```

---

### Task 3: Minimal GGUF reader/writer

**Files:**
- Create: `tools/gguf_lite.py`, `tools/test_gguf_lite.py`

**Interfaces:**
- Produces:
  - `class Reader(path)`: `.kv` (dict key → `(vtype, value)`, arrays as `(9, (elem_type, list))`), `.tensors` (list of dicts `name, dims, type, offset, nbytes, abs`), `.alignment`, `.data_start`, `.size`.
  - `nbytes(type, dims) -> int` (raises `ValueError` for an unknown type).
  - `write(path, kv, tensors, alignment) -> dict name → sha256 hex`; each tensor dict has `name, dims, type, nbytes` and either `src=(path, abs_offset)` or `data=bytes`.
  - Constants `T_U32=4, T_STR=8, T_ARR=9, T_U64=10` and `BLOCK` (type → (elems, bytes)).

- [ ] **Step 1: Write the failing tests** `tools/test_gguf_lite.py`

```python
import os
import pathlib
import tempfile
import unittest

import gguf_lite as g


class GgufLiteTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_round_trip(self):
        kv = {"general.architecture": (g.T_STR, "qwen4exp"),
              "qwen4exp.block_count": (g.T_U32, 48),
              "qwen4exp.attention.compress_ratios": (g.T_ARR, (g.T_U32, [0, 4] * 24)),
              "tokenizer.ggml.tokens": (g.T_ARR, (g.T_STR, ["a", "bc", ""]))}
        blob = bytes(range(256)) * 2
        tensors = [{"name": "t.q8", "dims": [32, 4], "type": 8, "nbytes": 4 * 34, "data": blob[:136]},
                   {"name": "t.f32", "dims": [3], "type": 0, "nbytes": 12, "data": blob[:12]}]
        shas = g.write(self.p / "a.gguf", kv, tensors, 32)
        r = g.Reader(self.p / "a.gguf")
        self.assertEqual(r.kv, kv)
        self.assertEqual([t["name"] for t in r.tensors], ["t.q8", "t.f32"])
        self.assertEqual([t["offset"] % 32 for t in r.tensors], [0, 0])
        with open(self.p / "a.gguf", "rb") as f:
            f.seek(r.tensors[1]["abs"])
            self.assertEqual(f.read(12), blob[:12])
        self.assertEqual(set(shas), {"t.q8", "t.f32"})

    def test_copy_from_source(self):
        src = self.p / "src.bin"
        src.write_bytes(b"x" * 100 + bytes(range(64)))
        t = [{"name": "c", "dims": [16], "type": 0, "nbytes": 64, "src": (str(src), 100)}]
        g.write(self.p / "b.gguf", {}, t, 32)
        r = g.Reader(self.p / "b.gguf")
        with open(self.p / "b.gguf", "rb") as f:
            f.seek(r.tensors[0]["abs"])
            self.assertEqual(f.read(64), bytes(range(64)))

    def test_nbytes(self):
        self.assertEqual(g.nbytes(21, [2560, 640, 512]), 2560 // 256 * 110 * 640 * 512)
        self.assertEqual(g.nbytes(20, [640, 2560]), 640 // 32 * 18 * 2560)
        self.assertEqual(g.nbytes(42, [640, 2560]), 640 // 64 * 18 * 2560)
        with self.assertRaises(ValueError):
            g.nbytes(99, [32])

    def test_not_gguf(self):
        (self.p / "x").write_bytes(b"NOPE" + bytes(20))
        with self.assertRaises(ValueError):
            g.Reader(self.p / "x")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python3 -m unittest test_gguf_lite -v 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'gguf_lite'`.

- [ ] **Step 3: Implement** `tools/gguf_lite.py`

```python
"""Minimal GGUF v3 reader/writer for ds4's repack tools: metadata and tensor infos are parsed, tensor data
is copied by offset (never interpreted)."""
import hashlib
import os
import struct

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 8: (32, 34), 10: (256, 84), 12: (256, 144),
         13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 20: (32, 18),
         21: (256, 110), 22: (256, 82), 23: (256, 136), 30: (1, 2), 39: (32, 17), 42: (64, 18)}
CHUNK = 64 << 20


def nbytes(ttype, dims):
    if ttype not in BLOCK:
        raise ValueError("unknown tensor type %d" % ttype)
    elems, size = BLOCK[ttype]
    if dims[0] % elems:
        raise ValueError("row of %d is not a whole number of %d-blocks" % (dims[0], elems))
    n = dims[0] // elems * size
    for d in dims[1:]:
        n *= d
    return n


def _align(n, a):
    return (n + a - 1) // a * a


class Reader:
    def __init__(self, path):
        self.path = str(path)
        self.size = os.path.getsize(self.path)
        with open(self.path, "rb") as f:
            self._f = f
            if f.read(4) != b"GGUF":
                raise ValueError("%s: not a GGUF file" % self.path)
            self.version = self._u("I")
            n_t, n_kv = self._u("Q"), self._u("Q")
            self.kv = {}
            for _ in range(n_kv):
                key = self._str()
                vt = self._u("I")
                self.kv[key] = (vt, self._val(vt))
            self.tensors = []
            for _ in range(n_t):
                name = self._str()
                dims = [self._u("Q") for _ in range(self._u("I"))]
                ttype, off = self._u("I"), self._u("Q")
                self.tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
            self.alignment = self.kv.get("general.alignment", (T_U32, 32))[1]
            self.data_start = _align(f.tell(), self.alignment)
        for t in self.tensors:
            t["nbytes"] = nbytes(t["type"], t["dims"])
            t["abs"] = self.data_start + t["offset"]
            if t["abs"] + t["nbytes"] > self.size:
                raise ValueError("%s: tensor %s ends past the file (%d > %d)" % (
                    self.path, t["name"], t["abs"] + t["nbytes"], self.size))

    def _u(self, fmt):
        return struct.unpack("<" + fmt, self._f.read(struct.calcsize("<" + fmt)))[0]

    def _str(self):
        return self._f.read(self._u("Q")).decode("utf-8")

    def _val(self, vt):
        if vt in _SCALAR:
            return self._u(_SCALAR[vt])
        if vt == T_STR:
            return self._str()
        if vt == T_ARR:
            et, n = self._u("I"), self._u("Q")
            return (et, [self._val(et) for _ in range(n)])
        raise ValueError("unknown metadata type %d" % vt)


def _pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_val(vt, v):
    if vt in _SCALAR:
        return struct.pack("<" + _SCALAR[vt], v)
    if vt == T_STR:
        return _pack_str(v)
    if vt == T_ARR:
        et, items = v
        return struct.pack("<IQ", et, len(items)) + b"".join(_pack_val(et, x) for x in items)
    raise ValueError("unknown metadata type %d" % vt)


def write(path, kv, tensors, alignment):
    """Writes a GGUF v3 file; returns {tensor name: sha256 hex of its data}."""
    head = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(kv))]
    for key, (vt, v) in kv.items():
        head += [_pack_str(key), struct.pack("<I", vt), _pack_val(vt, v)]
    off = 0
    for t in tensors:
        t["offset"] = off
        head += [_pack_str(t["name"]), struct.pack("<I", len(t["dims"])),
                 b"".join(struct.pack("<Q", d) for d in t["dims"]), struct.pack("<IQ", t["type"], off)]
        off = _align(off + t["nbytes"], alignment)
    header = b"".join(head)
    shas = {}
    with open(path, "wb") as out:
        out.write(header + b"\0" * (_align(len(header), alignment) - len(header)))
        for t in tensors:
            h = hashlib.sha256()
            if "data" in t:
                if len(t["data"]) != t["nbytes"]:
                    raise ValueError("%s: %d bytes of data for %d" % (t["name"], len(t["data"]), t["nbytes"]))
                out.write(t["data"])
                h.update(t["data"])
            else:
                src, at = t["src"]
                left = t["nbytes"]
                with open(src, "rb") as f:
                    f.seek(at)
                    while left:
                        buf = f.read(min(CHUNK, left))
                        if not buf:
                            raise ValueError("%s: source %s ends early" % (t["name"], src))
                        out.write(buf)
                        h.update(buf)
                        left -= len(buf)
            shas[t["name"]] = h.hexdigest()
            pad = _align(t["nbytes"], alignment) - t["nbytes"]
            out.write(b"\0" * pad)
    return shas
```

- [ ] **Step 4: Run the tests**

Run: `cd tools && python3 -m unittest test_gguf_lite -v 2>&1 | tail -3`
Expected: `Ran 4 tests ... OK`.

- [ ] **Step 5: Commit**

```bash
git add tools/gguf_lite.py tools/test_gguf_lite.py
git commit -m "tools: minimal GGUF v3 reader/writer for the repack"
```

---

### Task 4: CPU row dequantizers, fixtures, loader acceptance

**Files:**
- Modify: `ds4_quants.h` (the dequantizers), `ds4.c` (`ds4_dequant_row`, `qwen4_ref_row` cases ~l.70018, `qwen4_type_is_gsq`, `tensor_type_is_qwen4_dense` ~l.5595), `ds4.h`
- Create: `tools/gen_quant_fixtures.py`, generated `tests/quant_fixtures.h`
- Modify: `Makefile` (`ds4_test.o:` l.543 prerequisites + `tests/quant_fixtures.h`)
- Test: `tests/ds4_test.c` (`test_quant_dequant`, entry `--quant-dequant`)

**Interfaces:**
- Consumes: Task 1 enum, Task 2 tables, Task 3 `gguf_lite.Reader`.
- Produces: `int ds4_dequant_row(uint32_t type, const void *src, uint64_t n, float *out);` for types 13, 14, 16, 17, 18, 20, 21, 22, 23, 42 (0 = ok; -1 unknown type or `n` not whole blocks); `static bool qwen4_type_is_gsq(uint32_t type)` in ds4.c (true for 13, 14, 17, 18, 20, 21, 22, 23, 42); `tests/quant_fixtures.h` with `ds4_quant_fixture quant_fixtures[]` (`type, n, bytes, n_bytes, want`).

- [ ] **Step 1: Generate the fixtures** (reference values come from outside ds4)

Environment (outside git, once):

```bash
python3 -m venv ~/orca/workspaces/ds4-metal-data/sp3/venv
~/orca/workspaces/ds4-metal-data/sp3/venv/bin/pip install 'gguf>=0.10' numpy
```

`tools/gen_quant_fixtures.py`:

```python
"""Fixtures for ds4's quant row tests: two real blocks per type from a GGUF, with the values ggml's
reference dequantizers give (gguf-py quants.dequantize; Q2_0 from ggml's dequantize_row_q2_0).

usage: VENV_PYTHON tools/gen_quant_fixtures.py ISTA_SHARD1.gguf > tests/quant_fixtures.h"""
import hashlib
import pathlib
import sys

import numpy as np
import gguf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_lite  # noqa: E402

TYPES = [13, 14, 16, 17, 18, 20, 21, 22, 23, 42]


def q2_0_ref(raw):
    """ggml-quants.c dequantize_row_q2_0 @931351ea: y = ((q & 3) - 1) * d, four weights per byte."""
    out = []
    for i in range(0, len(raw), 18):
        d = np.frombuffer(raw[i:i + 2], dtype=np.float16)[0].astype(np.float32)
        qs = np.frombuffer(raw[i + 2:i + 18], dtype=np.uint8)
        q = np.stack([(qs >> (2 * k)) & 3 for k in range(4)], axis=1).reshape(-1).astype(np.int32)
        out.append((q - 1).astype(np.float32) * d)
    return np.concatenate(out)


def blocks(r, f, ttype):
    t = next(t for t in r.tensors if t["type"] == ttype)
    elems, size = gguf_lite.BLOCK[ttype]
    row_bytes = t["dims"][0] // elems * size
    rows = t["nbytes"] // row_bytes
    picks = [t["abs"], t["abs"] + (rows // 2) * row_bytes + (row_bytes // size // 2) * size]
    raw = b""
    for at in picks:
        f.seek(at)
        raw += f.read(size)
    return t["name"], raw


def main(argv):
    r = gguf_lite.Reader(argv[1])
    print("/* Generated by tools/gen_quant_fixtures.py: two blocks per type from")
    print(" * %s (%s)," % (pathlib.Path(argv[1]).name, "header sha256 %s" %
                           hashlib.sha256(open(argv[1], "rb").read(1 << 20)).hexdigest()[:16]))
    print(" * values from gguf-py %s quants.dequantize (Q2_0: ggml dequantize_row_q2_0). */" % gguf.__version__)
    print("typedef struct { uint32_t type, n; const uint8_t *bytes; uint32_t n_bytes; const float *want; } ds4_quant_fixture;")
    entries = []
    with open(argv[1], "rb") as f:
        for ttype in TYPES:
            name, raw = blocks(r, f, ttype)
            want = q2_0_ref(raw) if ttype == 42 else np.asarray(
                gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), gguf.GGMLQuantizationType(ttype)),
                dtype=np.float32).reshape(-1)
            print("/* %s */" % name)
            print("static const uint8_t qf_bytes_%d[] = {%s};" % (ttype, ",".join(str(b) for b in raw)))
            print("static const float qf_want_%d[] = {%s};" % (
                ttype, ",".join("%.9ef" % float(v) for v in want)))
            entries.append("    {%d, %d, qf_bytes_%d, %d, qf_want_%d}," % (ttype, len(want), ttype, len(raw), ttype))
    print("static const ds4_quant_fixture quant_fixtures[] = {\n%s\n};" % "\n".join(entries))


if __name__ == "__main__":
    main(sys.argv)
```

Run (reads a few KB of the ISTA file, fine during the SP4 batch):

```bash
~/orca/workspaces/ds4-metal-data/sp3/venv/bin/python tools/gen_quant_fixtures.py \
  ~/orca/workspaces/ds4-metal-data/gguf/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf > tests/quant_fixtures.h
```
Expected: `grep -c qf_bytes_ tests/quant_fixtures.h` = 20; the file is under 200 KB. If `gguf.GGMLQuantizationType(ttype)` rejects a type id, stop: the pip version is too old; pin a newer `gguf`.

- [ ] **Step 2: Write the failing test** (above `test_quant_types` in `tests/ds4_test.c`, with `#include "quant_fixtures.h"` after the existing includes)

```c
static void test_quant_dequant(void) {
    for (size_t i = 0; i < sizeof(quant_fixtures) / sizeof(quant_fixtures[0]); i++) {
        const ds4_quant_fixture *f = &quant_fixtures[i];
        float *got = calloc(f->n, sizeof(float));
        TEST_ASSERT(got != NULL);
        if (!got) return;
        TEST_ASSERT(ds4_dequant_row(f->type, f->bytes, f->n, got) == 0);
        uint32_t bad = 0;
        for (uint32_t j = 0; j < f->n; j++) {
            /* 2 ulp: -ffast-math may contract, and ggml's C and Python differ at that level */
            if (fabsf(got[j] - f->want[j]) > fabsf(f->want[j]) * 2.4e-7f) bad++;
        }
        if (bad) fprintf(stderr, "ds4-test: type %u: %u/%u values differ from ggml\n", f->type, bad, f->n);
        TEST_ASSERT(bad == 0);
        free(got);
    }
    float out[64];
    const uint8_t zero[18] = {0};
    TEST_ASSERT(ds4_dequant_row(42, zero, 63, out) != 0);   /* not a whole block */
    TEST_ASSERT(ds4_dequant_row(8, zero, 32, out) != 0);    /* Q8_0 is not in this table */
}
```

entry: `{"--quant-dequant", "quant-dequant", "GSQ-RCO row dequantizers match ggml (no model)", test_quant_dequant},` and in `ds4.h`:

```c
/* Dequantizes n weights (whole blocks) of a Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
 * IQ4_NL, IQ4_XS or Q2_0 row into out; returns 0, or -1 for another type or a partial block. */
int ds4_dequant_row(uint32_t type, const void *src, uint64_t n, float *out);
```

- [ ] **Step 3: Run it to verify it fails**

Run: `make ds4_test 2>&1 | tail -3`
Expected: link FAIL, `Undefined symbols ... _ds4_dequant_row`.

- [ ] **Step 4: Implement `ds4_quants.h`** (full content; ported from ggml-quants.c @931351ea)

```c
/* Row dequantizers for the GSQ-RCO quant types, ported from ggml-quants.c
 * (ggml-org/llama.cpp@931351ea, Copyright (c) 2023-2026 The ggml authors,
 * MIT License). Included by ds4.c after the IQ2 tables (kmask_iq2xs,
 * ksigns_iq2xs, iq2xxs_grid), ds4_iq_tables.h and f16_to_f32. n is a whole
 * number of blocks; every read is a byte read, so rows need no alignment. */
#pragma once

static float dq_half(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return f16_to_f32(h);
}

static void dq_iq2_xxs(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 66u) {
        const float d = dq_half(p);
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            uint16_t q2[4];
            memcpy(q2, p + 2 + ib32 * 8u, 8);
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            for (uint32_t j = 0; j < 4u; j++) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + ((aux_g >> (8u * j)) & 0xFFu));
                const uint32_t signs = ksigns_iq2xs[(aux_s >> (7u * j)) & 127u];
                for (uint32_t i = 0; i < 8u; i++) *y++ = dl * (float)grid[i] * (((signs >> i) & 1u) ? -1.0f : 1.0f);
            }
        }
    }
}

static void dq_iq2_xs(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 74u) {
        const float d = dq_half(p);
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            const uint8_t sc = p[66u + ib32];
            const float db[2] = { d * (0.5f + (float)(sc & 0xfu)) * 0.25f, d * (0.5f + (float)(sc >> 4)) * 0.25f };
            for (uint32_t l = 0; l < 4u; l++) {
                const uint32_t o = 2u + 2u * (4u * ib32 + l);
                const uint32_t q = (uint32_t)p[o] | ((uint32_t)p[o + 1u] << 8);
                const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q & 511u));
                const uint8_t signs = ksigns_iq2xs[q >> 9];
                for (uint32_t j = 0; j < 8u; j++) *y++ = db[l / 2u] * (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
        }
    }
}

static void dq_iq2_s(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 82u) {
        const float d = dq_half(p);
        const uint8_t *qs = p + 2u, *signs = p + 34u, *qh = p + 66u, *scales = p + 74u;
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            const float db[2] = { d * (0.5f + (float)(scales[ib32] & 0xfu)) * 0.25f,
                                  d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f };
            for (uint32_t l = 0; l < 4u; l++) {
                const uint8_t *grid = (const uint8_t *)(iq2s_grid + (qs[4u * ib32 + l] | ((qh[ib32] << (8u - 2u * l)) & 0x300u)));
                const uint8_t s = signs[4u * ib32 + l];
                for (uint32_t j = 0; j < 8u; j++) *y++ = db[l / 2u] * (float)grid[j] * ((s & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
        }
    }
}

static void dq_iq3_xxs(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 98u) {
        const float d = dq_half(p);
        const uint8_t *qs = p + 2u, *ss = p + 66u;
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, ss + 4u * ib32, 4);
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            for (uint32_t l = 0; l < 4u; l++, y += 8) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
                const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + qs[8u * ib32 + 2u * l]);
                const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + qs[8u * ib32 + 2u * l + 1u]);
                for (uint32_t j = 0; j < 4u; j++) {
                    y[j] = db * (float)g1[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    y[j + 4u] = db * (float)g2[j] * ((signs & kmask_iq2xs[j + 4u]) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

static void dq_iq3_s(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 110u) {
        const float d = dq_half(p);
        const uint8_t *qs = p + 2u, *qh = p + 66u, *signs = p + 74u, *scales = p + 106u;
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            const float db = d * (float)(1 + 2 * (int)((scales[ib32 / 2u] >> (4u * (ib32 & 1u))) & 0xfu));
            for (uint32_t l = 0; l < 4u; l++, y += 8) {
                const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[8u * ib32 + 2u * l] | ((qh[ib32] << (8u - 2u * l)) & 256u)));
                const uint8_t *g2 = (const uint8_t *)(iq3s_grid + (qs[8u * ib32 + 2u * l + 1u] | ((qh[ib32] << (7u - 2u * l)) & 256u)));
                const uint8_t s = signs[4u * ib32 + l];
                for (uint32_t j = 0; j < 4u; j++) {
                    y[j] = db * (float)g1[j] * ((s & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    y[j + 4u] = db * (float)g2[j] * ((s & kmask_iq2xs[j + 4u]) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

static void dq_iq4_nl(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 32u; b++, p += 18u, y += 32) {
        const float d = dq_half(p);
        for (uint32_t j = 0; j < 16u; j++) {
            y[j] = d * (float)kvalues_iq4nl[p[2u + j] & 0xfu];
            y[j + 16u] = d * (float)kvalues_iq4nl[p[2u + j] >> 4];
        }
    }
}

static void dq_iq4_xs(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 136u) {
        const float d = dq_half(p);
        const uint32_t scales_h = (uint32_t)p[2] | ((uint32_t)p[3] << 8);
        const uint8_t *qs = p + 8u;
        for (uint32_t ib = 0; ib < 8u; ib++, qs += 16, y += 32) {
            const int ls = (int)(((p[4u + ib / 2u] >> (4u * (ib % 2u))) & 0xfu) | (((scales_h >> (2u * ib)) & 3u) << 4));
            const float dl = d * (float)(ls - 32);
            for (uint32_t j = 0; j < 16u; j++) {
                y[j] = dl * (float)kvalues_iq4nl[qs[j] & 0xfu];
                y[j + 16u] = dl * (float)kvalues_iq4nl[qs[j] >> 4];
            }
        }
    }
}

static void dq_q2_0(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 64u; b++, p += 18u, y += 64) {
        const float d = dq_half(p);
        for (uint32_t j = 0; j < 64u; j++) {
            y[j] = (float)((int)((p[2u + j / 4u] >> ((j % 4u) * 2u)) & 3u) - 1) * d;
        }
    }
}

static void dq_q5_k(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 176u) {
        const float d = dq_half(p), dmin = dq_half(p + 2);
        const uint8_t *sc = p + 4u, *qh = p + 16u;
        for (uint32_t g = 0; g < 8u; g++) {
            uint32_t s, m;
            if (g < 4u) { s = sc[g] & 63u; m = sc[g + 4u] & 63u; }
            else { s = (sc[g + 4u] & 0xfu) | ((sc[g - 4u] >> 6) << 4); m = (sc[g + 4u] >> 4) | ((sc[g] >> 6) << 4); }
            const float dl = d * (float)s, ml = dmin * (float)m;
            const uint8_t *ql = p + 48u + 32u * (g / 2u);
            for (uint32_t l = 0; l < 32u; l++) {
                const uint32_t lo = (g & 1u) ? (uint32_t)(ql[l] >> 4) : (uint32_t)(ql[l] & 0xfu);
                *y++ = dl * (float)(lo + (((qh[l] >> g) & 1u) ? 16u : 0u)) - ml;
            }
        }
    }
}

static void dq_q6_k(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 210u, y += 256) {
        const float d = dq_half(p + 208u);
        for (uint32_t h = 0; h < 2u; h++) {
            const uint8_t *ql = p + 64u * h, *qh = p + 128u + 32u * h;
            const int8_t *sc = (const int8_t *)(p + 192u + 8u * h);
            float *yy = y + 128u * h;
            for (uint32_t l = 0; l < 32u; l++) {
                const uint32_t is = l / 16u;
                const int q1 = (int)((ql[l] & 0xfu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                const int q2 = (int)((ql[l + 32u] & 0xfu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                const int q4 = (int)((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                yy[l] = d * (float)sc[is] * (float)q1;
                yy[l + 32u] = d * (float)sc[is + 2u] * (float)q2;
                yy[l + 64u] = d * (float)sc[is + 4u] * (float)q3;
                yy[l + 96u] = d * (float)sc[is + 6u] * (float)q4;
            }
        }
    }
}
```

`ds4.c`: add `#include "ds4_quants.h"` just above `qwen4_ref_row` (~l.70018; by then `f16_to_f32`, the tables and `tensor_type` exist), and below it:

```c
int ds4_dequant_row(uint32_t type, const void *src, uint64_t n, float *out) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0 || (n % info->block_elems) != 0) return -1;
    const uint8_t *p = src;
    switch (type) {
    case DS4_TENSOR_Q5_K:    dq_q5_k(p, n, out); return 0;
    case DS4_TENSOR_Q6_K:    dq_q6_k(p, n, out); return 0;
    case DS4_TENSOR_IQ2_XXS: dq_iq2_xxs(p, n, out); return 0;
    case DS4_TENSOR_IQ2_XS:  dq_iq2_xs(p, n, out); return 0;
    case DS4_TENSOR_IQ2_S:   dq_iq2_s(p, n, out); return 0;
    case DS4_TENSOR_IQ3_XXS: dq_iq3_xxs(p, n, out); return 0;
    case DS4_TENSOR_IQ3_S:   dq_iq3_s(p, n, out); return 0;
    case DS4_TENSOR_IQ4_NL:  dq_iq4_nl(p, n, out); return 0;
    case DS4_TENSOR_IQ4_XS:  dq_iq4_xs(p, n, out); return 0;
    case DS4_TENSOR_Q2_0:    dq_q2_0(p, n, out); return 0;
    default:                 return -1;
    }
}
```

In `qwen4_ref_row`, replace the whole `case DS4_TENSOR_IQ2_XXS: {...}` body and add the new types, one path:

```c
    case DS4_TENSOR_IQ2_XXS: case DS4_TENSOR_Q5_K: case DS4_TENSOR_Q6_K:
    case DS4_TENSOR_IQ2_XS: case DS4_TENSOR_IQ2_S: case DS4_TENSOR_IQ3_XXS: case DS4_TENSOR_IQ3_S:
    case DS4_TENSOR_IQ4_NL: case DS4_TENSOR_IQ4_XS: case DS4_TENSOR_Q2_0: {
        const gguf_type_info *info = tensor_type(t->type);
        const uint64_t row_bytes = n / info->block_elems * info->block_bytes;
        if (ds4_dequant_row(t->type, (const uint8_t *)tensor_data(m, t) + row * row_bytes, n, out) != 0) {
            ds4_die("qwen4 reference: row is not a whole number of quant blocks");
        }
        break;
    }
```

Loader acceptance, after `tensor_type()`:

```c
/* The GSQ-RCO types: CPU rows (ds4_quants.h) and the Metal qwen4_row_dot. */
static bool qwen4_type_is_gsq(uint32_t type) {
    return type == DS4_TENSOR_Q5_K || type == DS4_TENSOR_Q6_K || type == DS4_TENSOR_IQ2_XS ||
           type == DS4_TENSOR_IQ2_S || type == DS4_TENSOR_IQ3_XXS || type == DS4_TENSOR_IQ3_S ||
           type == DS4_TENSOR_IQ4_NL || type == DS4_TENSOR_IQ4_XS || type == DS4_TENSOR_Q2_0;
}
```

and `tensor_type_is_qwen4_dense` returns `... || type == DS4_TENSOR_Q4_K || qwen4_type_is_gsq(type);` (update its comment: "plus the GSQ-RCO types").

Makefile l.543: append ` tests/quant_fixtures.h` to the `ds4_test.o:` prerequisites.

- [ ] **Step 5: Run the tests**

Run: `make ds4_test >/dev/null && ./ds4_test --quant-dequant --quant-types --qwen-yarn-policy --server`
Expected: all four `OK`, `ds4 tests: ok`. A mismatch names the type: fix the port (compare line by line with the upstream function), never the fixture.

- [ ] **Step 6: Commit**

```bash
git add ds4_quants.h ds4.c ds4.h tools/gen_quant_fixtures.py tests/quant_fixtures.h tests/ds4_test.c Makefile
git commit -m "ds4: CPU rows for the GSQ-RCO quant types, checked against ggml"
```

---

### Task 5: Repack tool

**Files:**
- Create: `tools/repack_ista_qwen4.py`, `tools/test_repack_ista_qwen4.py`

**Interfaces:**
- Consumes: `gguf_lite.Reader`, `gguf_lite.write`, `gguf_lite.BLOCK`, constants.
- Produces: CLI `python3 tools/repack_ista_qwen4.py ISTA_SHARD1 IVAN_GGUF OUT_GGUF` writing OUT and `OUT.json` (manifest); functions `plan(ista, ivan) -> (kv, tensors, conversions)` and `convert(raw_bf16, name) -> (type, bytes)`.

- [ ] **Step 1: Write the failing tests** `tools/test_repack_ista_qwen4.py`

```python
import json
import pathlib
import struct
import tempfile
import unittest

import numpy as np

import gguf_lite as g
import repack_ista_qwen4 as rp


def bf16(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


class RepackTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.dir.name)
        ista_kv = {"general.architecture": (g.T_STR, "qwen4exp"), "qwen4exp.block_count": (g.T_U32, 2),
                   "qwen4exp.attention.compress_ratios": (g.T_ARR, (g.T_U32, [0, 4])),
                   "split.count": (g.T_U16, 2), "split.no": (g.T_U16, 0), "split.tensors.count": (g.T_I32, 5)}
        self.hc = bf16([0.5, -2.0, 1.5, 0.25] * 8)
        ista_t = [
            {"name": "blk.0.ffn_down_exps.weight", "dims": [64, 2, 1], "type": 42, "nbytes": 36, "data": bytes(range(36))},
            {"name": "blk.0.ffn_gate_inp.weight", "dims": [32], "type": 30, "nbytes": 64, "data": bf16([1.0] * 32)},
            {"name": "blk.0.hc_attn_up.weight", "dims": [32], "type": 30, "nbytes": 64, "data": self.hc},
            {"name": "blk.0.hc_ffn_inject.weight", "dims": [32], "type": 30, "nbytes": 64,
             "data": bf16([1e-7] + [1.0] * 31)},   # 1e-7 is subnormal in F16: not exact
            {"name": "output_hc_down.weight", "dims": [32], "type": 30, "nbytes": 64, "data": bf16([3.0] * 32)},
        ]
        g.write(self.p / "ista.gguf", ista_kv, ista_t, 32)
        ivan_kv = {"general.architecture": (g.T_STR, "qwen4exp"), "qwen4exp.block_count": (g.T_U32, 3),
                   "qwen4exp.attention.compress_ratios": (g.T_ARR, (g.T_U32, [0, 4, 4])),
                   "general.alignment": (g.T_U32, 32), "qwen4exp.nextn_predict_layers": (g.T_U32, 1),
                   "qwen4exp.vocab_size": (g.T_U32, 248320), "qwen4exp.ple.seed": (g.T_U32, 1234),
                   "qwen4exp.ple.vocab_base": (g.T_U32, 20000000), "qwen4exp.ple.vocab_divisor": (g.T_U32, 128),
                   "qwen4exp.ple.row_count": (g.T_U64, 320001536), "qwen4exp.ple.row_dimension": (g.T_U32, 160),
                   "ds4.iq2.imatrix_sha256": (g.T_STR, "x")}
        ivan_t = [{"name": "blk.1.ffn_down_exps.weight", "dims": [32], "type": 0, "nbytes": 128, "data": b"\0" * 128},
                  {"name": "blk.2.nextn.eh_proj.weight", "dims": [32, 2], "type": 8, "nbytes": 68, "data": bytes(68)}]
        g.write(self.p / "ivan.gguf", ivan_kv, ivan_t, 32)

    def tearDown(self):
        self.dir.cleanup()

    def repack(self):
        rp.main(["x", str(self.p / "ista.gguf"), str(self.p / "ivan.gguf"), str(self.p / "out.gguf")])
        return g.Reader(self.p / "out.gguf"), json.loads((self.p / "out.gguf.json").read_text())

    def test_metadata(self):
        r, _ = self.repack()
        self.assertEqual(r.kv["qwen4exp.block_count"], (g.T_U32, 3))
        self.assertEqual(r.kv["qwen4exp.attention.compress_ratios"][1], (g.T_U32, [0, 4, 4]))
        self.assertEqual(r.kv["ds4.qwen4.down.logical_input"], (g.T_U32, 640))
        self.assertEqual(r.kv["ds4.qwen4.down.physical_input"], (g.T_U32, 640))
        for k in ("general.alignment", "qwen4exp.nextn_predict_layers", "qwen4exp.vocab_size",
                  "qwen4exp.ple.row_count", "qwen4exp.ple.seed"):
            self.assertIn(k, r.kv)
        for k in ("split.count", "split.no", "split.tensors.count", "ds4.iq2.imatrix_sha256"):
            self.assertNotIn(k, r.kv)

    def test_tensors_copied_and_grafted(self):
        r, man = self.repack()
        names = [t["name"] for t in r.tensors]
        self.assertIn("blk.2.nextn.eh_proj.weight", names)            # the MTP layer (index >= ISTA's count)
        self.assertNotIn("blk.1.ffn_down_exps.weight", names)         # Ivan's trunk is not taken
        t = next(t for t in r.tensors if t["name"] == "blk.0.ffn_down_exps.weight")
        with open(self.p / "out.gguf", "rb") as f:
            f.seek(t["abs"])
            self.assertEqual(f.read(36), bytes(range(36)))
        self.assertEqual(man["tensors"]["blk.0.ffn_down_exps.weight"]["converted_from"], None)

    def test_router_to_f32(self):
        r, man = self.repack()
        t = next(t for t in r.tensors if t["name"] == "blk.0.ffn_gate_inp.weight")
        self.assertEqual(t["type"], 0)
        with open(self.p / "out.gguf", "rb") as f:
            f.seek(t["abs"])
            self.assertEqual(struct.unpack("<32f", f.read(128)), (1.0,) * 32)
        self.assertEqual(man["tensors"]["blk.0.ffn_gate_inp.weight"]["converted_from"], "BF16")

    def test_hc_to_f16_when_exact(self):
        r, _ = self.repack()
        t = next(t for t in r.tensors if t["name"] == "blk.0.hc_attn_up.weight")
        self.assertEqual(t["type"], 1)
        with open(self.p / "out.gguf", "rb") as f:
            f.seek(t["abs"])
            self.assertEqual(list(np.frombuffer(f.read(64), dtype=np.float16)), [0.5, -2.0, 1.5, 0.25] * 8)

    def test_hc_falls_back_to_f32(self):
        r, _ = self.repack()
        t = next(t for t in r.tensors if t["name"] == "blk.0.hc_ffn_inject.weight")
        self.assertEqual(t["type"], 0)

    def test_other_bf16_untouched(self):
        r, _ = self.repack()
        self.assertEqual(next(t for t in r.tensors if t["name"] == "output_hc_down.weight")["type"], 30)

    def test_truncated_source_refused(self):
        data = (self.p / "ista.gguf").read_bytes()
        (self.p / "ista.gguf").write_bytes(data[:-40])
        with self.assertRaises(ValueError):
            self.repack()
        self.assertFalse((self.p / "out.gguf").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python3 -m unittest test_repack_ista_qwen4 -v 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'repack_ista_qwen4'`.

- [ ] **Step 3: Implement** `tools/repack_ista_qwen4.py`

```python
"""Repack ISTA-DASLab's GSQ-RCO Qwen3.8-Flash-Next shard 1 for ds4, grafting the MTP layer (blk.N, N = ISTA's
block_count) and the ds4 metadata from Ivan's ds4 GGUF.

usage: python3 tools/repack_ista_qwen4.py ISTA_SHARD1.gguf IVAN.gguf OUT.gguf   (writes OUT and OUT.json)"""
import hashlib
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
TO_F16 = re.compile(r"^blk\.\d+\.hc_(attn|ffn)_(up|inject)\.weight$")      # the graph takes F16/F32/Q8_0
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


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(64 << 20), b""):
            h.update(buf)
    return h.hexdigest()


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
```

The manifest's per-source sha256 is left out on purpose: hashing 47 GB twice more doubles the run; Task 6 records the known ISTA sha instead.

- [ ] **Step 4: Run the tests**

Run: `cd tools && python3 -m unittest test_repack_ista_qwen4 test_gguf_lite -v 2>&1 | tail -3`
Expected: `Ran 11 tests ... OK`.

- [ ] **Step 5: Commit**

```bash
git add tools/repack_ista_qwen4.py tools/test_repack_ista_qwen4.py
git commit -m "tools: repack ISTA GSQ-RCO shard 1 with Ivan's MTP layer for ds4"
```

---

### Task 6: Repack the real files (after the SP4 batch; disk-heavy, no GPU)

**Files:** none in git; output in `~/orca/workspaces/ds4-metal-data/gguf/ista/`.

**Interfaces:**
- Consumes: Task 5 CLI; Task 4 loader acceptance.
- Produces: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-DS4-MTP.gguf` (+ `.json`), used by Tasks 8-9.

- [ ] **Step 1: Wait for the SP4 batch to finish** (`~/orca/workspaces/ds4-metal-data/sp4/run-night.out` shows `== gate exit`). The repack reads 47 GB and writes 45 GB; running it during the batch would slow the SSD-streamed model under test.

- [ ] **Step 2: Run the repack**

```bash
D=~/orca/workspaces/ds4-metal-data/gguf
python3 tools/repack_ista_qwen4.py $D/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  $D/ivan/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf \
  $D/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-DS4-MTP.gguf
```
Expected: `wrote ...: 1255 tensors, 288 converted, 45.x GiB` (48 layers × (2 router + 4 hc) = 288).

- [ ] **Step 3: Verify every copied tensor against its source**

`~/orca/workspaces/ds4-metal-data/sp3/verify_repack.py` (throwaway, not in git):

```python
"""Every tensor of the repacked GGUF against its source: copies byte-equal (sha256 of the output bytes,
the source bytes and the manifest agree), conversions value-equal to the BF16 source."""
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/Users/dongnh/orca/workspaces/ds4-metal/sp3/tools")
import gguf_lite as g  # noqa: E402


def sha(path, at, n):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(at)
        while n:
            b = f.read(min(64 << 20, n))
            h.update(b)
            n -= len(b)
    return h.hexdigest()


def read(path, at, n):
    with open(path, "rb") as f:
        f.seek(at)
        return f.read(n)


out, ista, ivan = (g.Reader(p) for p in sys.argv[1:4])
man = json.loads(pathlib.Path(sys.argv[1] + ".json").read_text())["tensors"]
src = {t["name"]: (ista.path, t) for t in ista.tensors}
src.update({t["name"]: (ivan.path, t) for t in ivan.tensors if t["name"] not in src})
copied = conv = bad = 0
for t in out.tensors:
    path, s = src[t["name"]]
    if man[t["name"]]["converted_from"] is None:
        o, i = sha(out.path, t["abs"], t["nbytes"]), sha(path, s["abs"], s["nbytes"])
        ok = o == i == man[t["name"]]["sha256"] and s["type"] == t["type"]
        copied += 1
    else:
        want = (np.frombuffer(read(path, s["abs"], s["nbytes"]), np.uint16).astype(np.uint32) << 16).view(np.float32)
        got = np.frombuffer(read(out.path, t["abs"], t["nbytes"]),
                            np.float16 if t["type"] == 1 else np.float32).astype(np.float32)
        ok = np.array_equal(got, want)
        conv += 1
    if not ok:
        bad += 1
        print("MISMATCH", t["name"])
print("verified %d copied, %d converted, %d mismatches" % (copied, conv, bad))
```

Run: `python3 ~/orca/workspaces/ds4-metal-data/sp3/verify_repack.py $D/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-DS4-MTP.gguf $D/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf $D/ivan/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`
Expected: `verified 967 copied, 288 converted, 0 mismatches` (1255 − 288 = 967).

- [ ] **Step 4: ds4 parses and validates the file without the GPU**

Run: `./ds4 -m $D/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-DS4-MTP.gguf --ple $D/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --cpu --inspect`
Expected: exit 0 with the model summary (block count 49, the GSQ-RCO types listed). A layout error names the tensor: fix the loader acceptance (Task 4) or add an exact conversion (Task 5) with a test first.

- [ ] **Step 5: Record** the output size, sha256 (`shasum -a 256`) and the verify line in the ledger (no commit: nothing in git changed).

---

### Task 7: Metal row dot for the GSQ-RCO types (GPU: minutes, needs the user's go-ahead)

**Files:**
- Modify: `metal/qwen4.metal` (lane helpers before `qwen4_row_dot` ~l.2821; branches inside it before `weight_type == 30`; comment on `weight_type` in `ds4_metal_args_qwen4_moe`)
- Modify: `ds4_metal.m` (`qwen4_expert_row_bytes` ~l.49922)
- Test: `tests/ds4_test.c` (`test_metal_qwen4_quant_gemv` in `test_metal_kernel_group`, `__APPLE__` block)

**Interfaces:**
- Consumes: Task 2 Metal tables; Task 4 `ds4_dequant_row`, `quant_fixtures`.
- Produces: `qwen4_row_dot` handles types 13, 14, 17, 18, 20, 21, 22, 23, 42; `qwen4_expert_row_bytes` sizes them (Q2_0 needs `in_dim % 64 == 0`).

- [ ] **Step 1: Write the failing test** (in `tests/ds4_test.c`, inside the `#if defined(__APPLE__)` block after `test_metal_q8_0_decode_pair_exact`)

```c
/* Rows of a GSQ-RCO type built from the fixture's two real blocks, varied per row. */
static void test_quant_fill_rows(uint8_t *dst, const ds4_quant_fixture *f, uint32_t in_dim, uint32_t rows) {
    uint32_t elems = 0, bytes = 0;
    (void)ds4_gguf_type_block(f->type, &elems, &bytes);
    const uint32_t per_row = in_dim / elems;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t k = 0; k < per_row; k++) {
            const uint32_t pick = (r * 7u + k * 3u) % 2u;
            memcpy(dst + ((uint64_t)r * per_row + k) * bytes, f->bytes + (uint64_t)pick * bytes, bytes);
        }
    }
}

static double test_quant_row_ref(uint32_t type, const uint8_t *row, uint32_t in_dim, const float *x, double *mag) {
    float *w = malloc((size_t)in_dim * sizeof(float));
    double acc = 0.0, m = 0.0;
    if (w && ds4_dequant_row(type, row, in_dim, w) == 0) {
        for (uint32_t i = 0; i < in_dim; i++) { acc += (double)w[i] * x[i]; m += fabs((double)w[i] * x[i]); }
    }
    free(w);
    if (mag) *mag = m;
    return acc;
}

static void test_quant_x(float *x, uint32_t n, uint32_t salt) {
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t k = i + salt * 131u;
        x[i] = (float)((int)((k * 29u + (k ^ (k >> 3u)) * 7u) % 127u) - 63) / 72.0f;
    }
}

static void test_metal_qwen4_quant_gemv(void) {
    const uint64_t page = (uint64_t)getpagesize();
    for (size_t i = 0; i < sizeof(quant_fixtures) / sizeof(quant_fixtures[0]); i++) {
        const ds4_quant_fixture *f = &quant_fixtures[i];
        if (f->type == 16) continue;                        /* IQ2_XXS already has its own kernels */
        const uint32_t dims[2] = { 2560u, (f->type == 20 || f->type == 42) ? 640u : 6144u };
        for (int di = 0; di < 2; di++) {
            const uint32_t in_dim = dims[di], rows = 37u, n_tok = 3u;
            uint32_t elems = 0, bytes = 0;
            (void)ds4_gguf_type_block(f->type, &elems, &bytes);
            const uint64_t row_bytes = (uint64_t)in_dim / elems * bytes;
            const uint64_t alloc = test_round_up_u64(row_bytes * rows, page);
            void *wraw = NULL;
            TEST_ASSERT(posix_memalign(&wraw, (size_t)page, (size_t)alloc) == 0);
            if (!wraw) return;
            test_quant_fill_rows(wraw, f, in_dim, rows);
            float *xh = malloc((size_t)n_tok * in_dim * sizeof(float));
            float *oh = malloc((size_t)n_tok * rows * sizeof(float));
            for (uint32_t t = 0; t < n_tok; t++) test_quant_x(xh + (uint64_t)t * in_dim, in_dim, t);
            ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
            ds4_gpu_tensor *o = ds4_gpu_tensor_alloc((uint64_t)n_tok * rows * sizeof(float));
            TEST_ASSERT(x && o && xh && oh);
            if (x && o && xh && oh) {
                TEST_ASSERT(ds4_gpu_tensor_write(x, 0, xh, (uint64_t)n_tok * in_dim * sizeof(float)) != 0);
                TEST_ASSERT(ds4_gpu_set_model_map(wraw, alloc) != 0);
                ds4_gpu_tensor *outs[1] = { o };
                const uint64_t offs[1] = { 0 };
                const uint32_t types[1] = { f->type }, out_rows[1] = { rows };
                TEST_ASSERT(ds4_gpu_qwen4_multi_gemv_tensor(x, n_tok, in_dim, 1, outs, wraw, alloc, offs, types, out_rows) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(o, 0, oh, (uint64_t)n_tok * rows * sizeof(float)) != 0);
                uint32_t bad = 0;
                for (uint32_t t = 0; t < n_tok; t++) {
                    for (uint32_t r = 0; r < rows; r++) {
                        double mag = 0.0;
                        const double ref = test_quant_row_ref(f->type, (const uint8_t *)wraw + r * row_bytes, in_dim,
                                                              xh + (uint64_t)t * in_dim, &mag);
                        if (fabs((double)oh[(uint64_t)t * rows + r] - ref) > 1e-5 * mag + 1e-6) bad++;
                    }
                }
                if (bad) fprintf(stderr, "ds4-test: Metal row dot type %u dim %u: %u/%u rows off\n",
                                 f->type, in_dim, bad, n_tok * rows);
                TEST_ASSERT(bad == 0);
            }
            ds4_gpu_tensor_free(x);
            ds4_gpu_tensor_free(o);
            free(xh);
            free(oh);
            free(wraw);
        }
    }
}
```

The MoE kernels (routed slots through the selected-expert offsets, 640-wide unpadded down rows, the
shared expert as an extra slot with its own gate/up type and its own down type) get their own case test:

```c
static const ds4_quant_fixture *test_quant_fixture(uint32_t type) {
    for (size_t i = 0; i < sizeof(quant_fixtures) / sizeof(quant_fixtures[0]); i++)
        if (quant_fixtures[i].type == type) return &quant_fixtures[i];
    return NULL;
}

static uint64_t test_quant_row_bytes(uint32_t type, uint32_t in_dim) {
    uint32_t e = 0, b = 0;
    if (ds4_gguf_type_block(type, &e, &b) != 0) return 0;
    return (uint64_t)in_dim / e * b;
}

/* Routed gate/up type gu and down type dn on 3 experts; shared gate/up type sg and shared down
 * type sd (UINT32_MAX: no shared expert). CPU reference from ds4_dequant_row, doubles. */
static void test_metal_qwen4_quant_moe_case(uint32_t gu, uint32_t dn, uint32_t sg, uint32_t sd) {
    const uint32_t in_dim = 2560u, ff = 640u, out_dim = 2560u, n_exp = 3u, T = 2u, slots = 2u;
    const bool has_sh = sg != UINT32_MAX;
    const ds4_quant_fixture *fg = test_quant_fixture(gu), *fd = test_quant_fixture(dn);
    const ds4_quant_fixture *fsg = has_sh ? test_quant_fixture(sg) : NULL, *fsd = has_sh ? test_quant_fixture(sd) : NULL;
    TEST_ASSERT(fg && fd && (!has_sh || (fsg && fsd)));
    if (!fg || !fd || (has_sh && (!fsg || !fsd))) return;
    const uint64_t g_row = test_quant_row_bytes(gu, in_dim), g_exp = g_row * ff;
    const uint64_t d_row = test_quant_row_bytes(dn, ff), d_exp = d_row * out_dim;
    const uint64_t sg_row = has_sh ? test_quant_row_bytes(sg, in_dim) : 0, sd_row = has_sh ? test_quant_row_bytes(sd, ff) : 0;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t off_up = test_round_up_u64(g_exp * n_exp, page);
    const uint64_t off_dn = off_up + test_round_up_u64(g_exp * n_exp, page);
    const uint64_t off_sg = off_dn + test_round_up_u64(d_exp * n_exp, page);
    const uint64_t off_su = off_sg + test_round_up_u64(sg_row * ff + 1u, page);
    const uint64_t off_sd = off_su + test_round_up_u64(sg_row * ff + 1u, page);
    const uint64_t alloc = off_sd + test_round_up_u64(sd_row * out_dim + 1u, page);
    uint8_t *w = NULL;
    TEST_ASSERT(posix_memalign((void **)&w, (size_t)page, (size_t)alloc) == 0);
    if (!w) return;
    memset(w, 0, (size_t)alloc);
    test_quant_fill_rows(w, fg, in_dim, ff * n_exp);
    test_quant_fill_rows(w + off_up, fg, in_dim, ff * n_exp);
    memcpy(w + off_up, w + off_up + g_row, (size_t)g_row);           /* up row 0 differs from gate row 0 */
    test_quant_fill_rows(w + off_dn, fd, ff, out_dim * n_exp);
    if (has_sh) {
        test_quant_fill_rows(w + off_sg, fsg, in_dim, ff);
        test_quant_fill_rows(w + off_su, fsg, in_dim, ff);
        test_quant_fill_rows(w + off_sd, fsd, ff, out_dim);
    }
    const int32_t sel[4] = { 0, 2, 1, 0 };
    const uint32_t n_out = slots + (has_sh ? 1u : 0u);
    float *xh = malloc((size_t)T * in_dim * sizeof(float));
    float *mh = malloc((size_t)T * n_out * ff * sizeof(float));
    float *ph = malloc((size_t)T * n_out * out_dim * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)T * in_dim * sizeof(float));
    ds4_gpu_tensor *s = ds4_gpu_tensor_alloc(sizeof(sel));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)T * n_out * ff * sizeof(float));
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc((uint64_t)T * n_out * out_dim * sizeof(float));
    TEST_ASSERT(xh && mh && ph && x && s && mid && part);
    if (xh && mh && ph && x && s && mid && part) {
        for (uint32_t t = 0; t < T; t++) test_quant_x(xh + (uint64_t)t * in_dim, in_dim, t + 5u);
        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, xh, (uint64_t)T * in_dim * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(s, 0, sel, sizeof(sel)) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(w, alloc) != 0);
        TEST_ASSERT(ds4_gpu_qwen4_moe_mid_tensor(mid, x, s, w, alloc, 0, off_up, gu, n_exp, T, slots, in_dim, ff,
                                                 off_sg, off_su, has_sh ? sg : UINT32_MAX) != 0);
        TEST_ASSERT(ds4_gpu_qwen4_moe_down_tensor(part, mid, s, w, alloc, off_dn, dn, n_exp, T, slots, ff, out_dim,
                                                  off_sd, has_sh ? sd : UINT32_MAX) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(mid, 0, mh, (uint64_t)T * n_out * ff * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(part, 0, ph, (uint64_t)T * n_out * out_dim * sizeof(float)) != 0);
        uint32_t bad_mid = 0, bad_down = 0;
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t k = 0; k < n_out; k++) {
                const bool shared = k == slots;
                const uint32_t ty = shared ? sg : gu, dty = shared ? sd : dn;
                const uint64_t row = shared ? sg_row : g_row, drow = shared ? sd_row : d_row;
                const uint64_t gbase = shared ? off_sg : (uint64_t)sel[t * slots + k] * g_exp;
                const uint64_t ubase = shared ? off_su : off_up + (uint64_t)sel[t * slots + k] * g_exp;
                const uint64_t dbase = shared ? off_sd : off_dn + (uint64_t)sel[t * slots + k] * d_exp;
                const float *xt = xh + (uint64_t)t * in_dim, *mt = mh + ((uint64_t)t * n_out + k) * ff;
                for (uint32_t r = 0; r < ff; r++) {
                    double mg = 0.0, mu = 0.0;
                    const double gv = test_quant_row_ref(ty, w + gbase + r * row, in_dim, xt, &mg);
                    const double uv = test_quant_row_ref(ty, w + ubase + r * row, in_dim, xt, &mu);
                    const double silu = gv / (1.0 + exp(-gv)), ref = silu * uv;
                    if (fabs((double)mt[r] - ref) > 2e-5 * (fabs(uv) * mg + fabs(silu) * mu) + 1e-6) bad_mid++;
                }
                for (uint32_t r = 0; r < out_dim; r++) {
                    double mag = 0.0;
                    const double ref = test_quant_row_ref(dty, w + dbase + r * drow, ff, mt, &mag);
                    if (fabs((double)ph[((uint64_t)t * n_out + k) * out_dim + r] - ref) > 1e-5 * mag + 1e-6) bad_down++;
                }
            }
        }
        if (bad_mid || bad_down) fprintf(stderr, "ds4-test: Qwen MoE %u/%u shared %u/%u: mid %u, down %u rows off\n",
                                         gu, dn, sg, sd, bad_mid, bad_down);
        TEST_ASSERT(bad_mid == 0 && bad_down == 0);
    }
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(s);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(part);
    free(xh);
    free(mh);
    free(ph);
    free(w);
}

static void test_metal_qwen4_quant_moe(void) {
    test_metal_qwen4_quant_moe_case(17, 42, UINT32_MAX, UINT32_MAX);   /* IQ2_XS gate/up, Q2_0 down */
    test_metal_qwen4_quant_moe_case(22, 20, UINT32_MAX, UINT32_MAX);   /* IQ2_S, IQ4_NL */
    test_metal_qwen4_quant_moe_case(18, 42, 21, 20);                   /* IQ3_XXS, Q2_0; shared IQ3_S / IQ4_NL */
    test_metal_qwen4_quant_moe_case(21, 20, 23, 42);                   /* IQ3_S, IQ4_NL; shared IQ4_XS / Q2_0 */
    test_metal_qwen4_quant_moe_case(16, 20, 23, 20);                   /* IQ2_XXS (M5 NR kernels), GSQ shared slot */
}
```

Call `test_metal_qwen4_quant_gemv();` and `test_metal_qwen4_quant_moe();` at the end of the `__APPLE__` block of `test_metal_kernel_group`.

- [ ] **Step 2: Build; ask the user for a short GPU window; run it to verify it fails**

Run: `make ds4_test >/dev/null && ./ds4_test --metal-kernels 2>&1 | grep -E "row dot|Qwen MoE|metal-kernels"`
Expected: FAIL: `Metal row dot type ...` lines for every new type, and the MoE cases fail (the host returns 0 for a zero row size, or the kernel's F32 fallback reads garbage).

- [ ] **Step 3: Implement the lane helpers** in `metal/qwen4.metal`, directly above `qwen4_row_dot`'s definition:

```metal
/* GSQ-RCO quant types (ggml-quants.c dequantizers @931351ea, MIT; the CPU
 * rows in ds4_quants.h). One simdgroup per row: for 256-weight super-blocks
 * lane t takes sub-block t % 8 of blocks t / 8, t / 8 + 4, ...; the 32/64
 * weight blocks go round-robin. Each returns the lane's partial sum. */
static inline float qwen4_lane_iq2_xs(device const uchar *row, device const float *x, uint in_dim, ushort tiisg) {
    const uint nb = in_dim / 256u, ib32 = tiisg % 8u;
    float acc = 0.0f;
    for (uint ib = tiisg / 8u; ib < nb; ib += 4u) {
        device const uchar *blk = row + (uint64_t)ib * 74u;
        const float d = (float)(*(device const half *)blk);
        const uint sc = blk[66u + ib32];
        const float db0 = d * (0.5f + (float)(sc & 0xfu)) * 0.25f, db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        device const ushort *qs = (device const ushort *)(blk + 2u) + 4u * ib32;
        device const float *y = x + ib * 256u + ib32 * 32u;
        for (uint l = 0; l < 4u; l++) {
            const uint q = qs[l];
            constant const uchar *grid = (constant const uchar *)(ds4_metal_iq2xs_grid + (q & 511u));
            const uint signs = ds4_metal_ksigns_iq2xs[q >> 9];
            float part = 0.0f;
            for (uint j = 0; j < 8u; j++) part += (float)grid[j] * (((signs >> j) & 1u) ? -y[l * 8u + j] : y[l * 8u + j]);
            acc += (l < 2u ? db0 : db1) * part;
        }
    }
    return acc;
}

static inline float qwen4_lane_iq2_s(device const uchar *row, device const float *x, uint in_dim, ushort tiisg) {
    const uint nb = in_dim / 256u, ib32 = tiisg % 8u;
    float acc = 0.0f;
    for (uint ib = tiisg / 8u; ib < nb; ib += 4u) {
        device const uchar *blk = row + (uint64_t)ib * 82u;
        const float d = (float)(*(device const half *)blk);
        device const uchar *qs = blk + 2u + 4u * ib32, *signs = blk + 34u + 4u * ib32;
        const uint qh = blk[66u + ib32], sc = blk[74u + ib32];
        const float db0 = d * (0.5f + (float)(sc & 0xfu)) * 0.25f, db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        device const float *y = x + ib * 256u + ib32 * 32u;
        for (uint l = 0; l < 4u; l++) {
            constant const uchar *grid = (constant const uchar *)(ds4_metal_iq2s_grid + ((uint)qs[l] | ((qh << (8u - 2u * l)) & 0x300u)));
            const uint s = signs[l];
            float part = 0.0f;
            for (uint j = 0; j < 8u; j++) part += (float)grid[j] * (((s >> j) & 1u) ? -y[l * 8u + j] : y[l * 8u + j]);
            acc += (l < 2u ? db0 : db1) * part;
        }
    }
    return acc;
}

static inline float qwen4_lane_iq3_xxs(device const uchar *row, device const float *x, uint in_dim, ushort tiisg) {
    const uint nb = in_dim / 256u, ib32 = tiisg % 8u;
    float acc = 0.0f;
    for (uint ib = tiisg / 8u; ib < nb; ib += 4u) {
        device const uchar *blk = row + (uint64_t)ib * 98u;
        const float d = (float)(*(device const half *)blk);
        device const uchar *qs = blk + 2u + 8u * ib32;
        device const ushort *ss = (device const ushort *)(blk + 66u) + 2u * ib32;
        const uint aux = (uint)ss[0] | ((uint)ss[1] << 16);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        device const float *y = x + ib * 256u + ib32 * 32u;
        for (uint l = 0; l < 4u; l++) {
            const uint signs = ds4_metal_ksigns_iq2xs[(aux >> (7u * l)) & 127u];
            constant const uchar *g1 = (constant const uchar *)(ds4_metal_iq3xxs_grid + qs[2u * l]);
            constant const uchar *g2 = (constant const uchar *)(ds4_metal_iq3xxs_grid + qs[2u * l + 1u]);
            float part = 0.0f;
            for (uint j = 0; j < 4u; j++) {
                part += (float)g1[j] * (((signs >> j) & 1u) ? -y[l * 8u + j] : y[l * 8u + j]);
                part += (float)g2[j] * (((signs >> (j + 4u)) & 1u) ? -y[l * 8u + 4u + j] : y[l * 8u + 4u + j]);
            }
            acc += db * part;
        }
    }
    return acc;
}

static inline float qwen4_lane_iq3_s(device const uchar *row, device const float *x, uint in_dim, ushort tiisg) {
    const uint nb = in_dim / 256u, ib32 = tiisg % 8u;
    float acc = 0.0f;
    for (uint ib = tiisg / 8u; ib < nb; ib += 4u) {
        device const uchar *blk = row + (uint64_t)ib * 110u;
        const float d = (float)(*(device const half *)blk);
        device const uchar *qs = blk + 2u + 8u * ib32, *signs = blk + 74u + 4u * ib32;
        const uint qh = blk[66u + ib32];
        const float db = d * (float)(1u + 2u * ((blk[106u + ib32 / 2u] >> (4u * (ib32 & 1u))) & 0xfu));
        device const float *y = x + ib * 256u + ib32 * 32u;
        for (uint l = 0; l < 4u; l++) {
            constant const uchar *g1 = (constant const uchar *)(ds4_metal_iq3s_grid + ((uint)qs[2u * l] | ((qh << (8u - 2u * l)) & 256u)));
            constant const uchar *g2 = (constant const uchar *)(ds4_metal_iq3s_grid + ((uint)qs[2u * l + 1u] | ((qh << (7u - 2u * l)) & 256u)));
            const uint s = signs[l];
            float part = 0.0f;
            for (uint j = 0; j < 4u; j++) {
                part += (float)g1[j] * (((s >> j) & 1u) ? -y[l * 8u + j] : y[l * 8u + j]);
                part += (float)g2[j] * (((s >> (j + 4u)) & 1u) ? -y[l * 8u + 4u + j] : y[l * 8u + 4u + j]);
            }
            acc += db * part;
        }
    }
    return acc;
}

static inline float qwen4_lane_iq4_nl(device const uchar *row, device const float *x, uint in_dim, ushort tiisg) {
    float acc = 0.0f;
    for (uint ib = tiisg; ib < in_dim / 32u; ib += 32u) {
        device const uchar *blk = row + (uint64_t)ib * 18u;
        device const float *y = x + ib * 32u;
        float part = 0.0f;
        for (uint j = 0; j < 16u; j++) {
            part += (float)ds4_metal_kvalues_iq4nl[blk[2u + j] & 0xfu] * y[j] +
                    (float)ds4_metal_kvalues_iq4nl[blk[2u + j] >> 4] * y[j + 16u];
        }
        acc += (float)(*(device const half *)blk) * part;
    }
    return acc;
}

static inline float qwen4_lane_iq4_xs(device const uchar *row, device const float *x, uint in_dim, ushort tiisg) {
    const uint nb = in_dim / 256u, ib32 = tiisg % 8u;
    float acc = 0.0f;
    for (uint ib = tiisg / 8u; ib < nb; ib += 4u) {
        device const uchar *blk = row + (uint64_t)ib * 136u;
        const uint scales_h = (uint)*(device const ushort *)(blk + 2u);
        const int ls = (int)(((blk[4u + ib32 / 2u] >> (4u * (ib32 % 2u))) & 0xfu) | (((scales_h >> (2u * ib32)) & 3u) << 4));
        device const uchar *qs = blk + 8u + 16u * ib32;
        device const float *y = x + ib * 256u + ib32 * 32u;
        float part = 0.0f;
        for (uint j = 0; j < 16u; j++) {
            part += (float)ds4_metal_kvalues_iq4nl[qs[j] & 0xfu] * y[j] +
                    (float)ds4_metal_kvalues_iq4nl[qs[j] >> 4] * y[j + 16u];
        }
        acc += (float)(*(device const half *)blk) * (float)(ls - 32) * part;
    }
    return acc;
}

static inline float qwen4_lane_q2_0(device const uchar *row, device const float *x, uint in_dim, ushort tiisg) {
    float acc = 0.0f;
    for (uint ib = tiisg; ib < in_dim / 64u; ib += 32u) {
        device const uchar *blk = row + (uint64_t)ib * 18u;
        device const float *y = x + ib * 64u;
        float part = 0.0f;
        for (uint j = 0; j < 64u; j++) part += (float)((int)((blk[2u + j / 4u] >> ((j % 4u) * 2u)) & 3u) - 1) * y[j];
        acc += (float)(*(device const half *)blk) * part;
    }
    return acc;
}

static inline float qwen4_lane_q5_k(device const uchar *row, device const float *x, uint in_dim, ushort tiisg) {
    const uint nb = in_dim / 256u, g = tiisg % 8u;
    float acc = 0.0f;
    for (uint ib = tiisg / 8u; ib < nb; ib += 4u) {
        device const uchar *blk = row + (uint64_t)ib * 176u;
        const float d = (float)(*(device const half *)blk), dmin = (float)(*(device const half *)(blk + 2u));
        device const uchar *sc = blk + 4u;
        uint s, m;
        if (g < 4u) { s = sc[g] & 63u; m = sc[g + 4u] & 63u; }
        else { s = (sc[g + 4u] & 0xfu) | ((sc[g - 4u] >> 6) << 4); m = (sc[g + 4u] >> 4) | ((sc[g] >> 6) << 4); }
        device const uchar *qh = blk + 16u, *ql = blk + 48u + 32u * (g / 2u);
        device const float *y = x + ib * 256u + g * 32u;
        float part = 0.0f, ysum = 0.0f;
        for (uint l = 0; l < 32u; l++) {
            const uint lo = (g & 1u) ? (uint)(ql[l] >> 4) : (uint)(ql[l] & 0xfu);
            part += (float)(lo + (((qh[l] >> g) & 1u) << 4)) * y[l];
            ysum += y[l];
        }
        acc += d * (float)s * part - dmin * (float)m * ysum;
    }
    return acc;
}

static inline float qwen4_lane_q6_k(device const uchar *row, device const float *x, uint in_dim, ushort tiisg) {
    const uint nb = in_dim / 256u, g = tiisg % 8u, h = g / 4u, k = g % 4u;
    float acc = 0.0f;
    for (uint ib = tiisg / 8u; ib < nb; ib += 4u) {
        device const uchar *blk = row + (uint64_t)ib * 210u;
        device const uchar *ql = blk + 64u * h + ((k & 1u) ? 32u : 0u), *qh = blk + 128u + 32u * h;
        device const char *sc = (device const char *)(blk + 192u + 8u * h);
        device const float *y = x + ib * 256u + 128u * h + 32u * k;
        float part0 = 0.0f, part1 = 0.0f;
        for (uint l = 0; l < 32u; l++) {
            const uint lo = (k < 2u) ? (uint)(ql[l] & 0xfu) : (uint)(ql[l] >> 4);
            const float q = (float)((int)(lo | (((qh[l] >> (2u * k)) & 3u) << 4)) - 32);
            if (l < 16u) part0 += q * y[l]; else part1 += q * y[l];
        }
        acc += (float)(*(device const half *)(blk + 208u)) * ((float)sc[2u * k] * part0 + (float)sc[2u * k + 1u] * part1);
    }
    return acc;
}
```

Inside `qwen4_row_dot`, before `} else if (weight_type == 30) {`:

```metal
    } else if (weight_type == 17) {
        acc = qwen4_lane_iq2_xs((device const uchar *)row, x, in_dim, tiisg);
    } else if (weight_type == 22) {
        acc = qwen4_lane_iq2_s((device const uchar *)row, x, in_dim, tiisg);
    } else if (weight_type == 18) {
        acc = qwen4_lane_iq3_xxs((device const uchar *)row, x, in_dim, tiisg);
    } else if (weight_type == 21) {
        acc = qwen4_lane_iq3_s((device const uchar *)row, x, in_dim, tiisg);
    } else if (weight_type == 20) {
        acc = qwen4_lane_iq4_nl((device const uchar *)row, x, in_dim, tiisg);
    } else if (weight_type == 23) {
        acc = qwen4_lane_iq4_xs((device const uchar *)row, x, in_dim, tiisg);
    } else if (weight_type == 42) {
        acc = qwen4_lane_q2_0((device const uchar *)row, x, in_dim, tiisg);
    } else if (weight_type == 13) {
        acc = qwen4_lane_q5_k((device const uchar *)row, x, in_dim, tiisg);
    } else if (weight_type == 14) {
        acc = qwen4_lane_q6_k((device const uchar *)row, x, in_dim, tiisg);
```

Update the `weight_type` comment in `ds4_metal_args_qwen4_moe` to list the new ids.

In `ds4_metal.m` `qwen4_expert_row_bytes`, add before `default`:

```c
    case 13u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 176u;   /* q5_K */
    case 14u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 210u;   /* q6_K */
    case 17u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 74u;    /* iq2_xs */
    case 18u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 98u;    /* iq3_xxs */
    case 20u: return (in_dim / 32u) * 18u;                            /* iq4_nl */
    case 21u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 110u;   /* iq3_s */
    case 22u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 82u;    /* iq2_s */
    case 23u: return (in_dim % 256u) ? 0u : (in_dim / 256u) * 136u;   /* iq4_xs */
    case 42u: return (in_dim % 64u) ? 0u : (in_dim / 64u) * 18u;      /* q2_0 */
```
(`qwen35_expert_row_bytes` handles 13 before calling this, so Ornith is unchanged.)

- [ ] **Step 4: Run the Metal group** (same GPU window)

Run: `make ds4_test >/dev/null && ./ds4_test --metal-kernels`
Expected: `metal-kernels: OK` (every existing Metal test still passes). Then the CPU groups: `./ds4_test --quant-types --quant-dequant --qwen-yarn-policy --qwen-kv-grow-policy --dir-steering-rows --server` → `ds4 tests: ok`.

- [ ] **Step 5: Commit**

```bash
git add metal/qwen4.metal ds4_metal.m tests/ds4_test.c
git commit -m "metal: qwen4 row dot for the GSQ-RCO quant types"
```

---

### Task 8: Graph wiring, dense routing, shared-expert types (GPU: minutes, needs Task 6's file)

**Files:**
- Modify: `ds4.c` (new `qwen4_graph_gsq_ok` above `qwen4_graph_dense_ok` ~l.59554; `qwen4_graph_dense_ok`; `qwen4_graph_expert_ok` ~l.59565; the Apple `switch (w->type)` in `qwen4_gemv_rows` ~l.60340; `shared_dense` in the MoE layer ~l.61121)
- Test: the model-level GPU-vs-CPU check (`DS4_QWEN4_GPU=1 ./ds4 --first-token-test`), `make cpu`

**Interfaces:**
- Consumes: Tasks 4, 6, 7.
- Produces: the qwen4 Metal graph accepts and runs models with GSQ-RCO dense and expert tensors, including layers whose shared gate and up differ in type.

**Why the shared-expert change:** below nine rows (decode, MTP verify) the MoE layer runs the shared expert as an extra slot of `kernel_qwen4_moe_mid`, which takes one `shared_type` for gate and up (`l->ffn_gate_shexp->type`). ISTA quantized them independently (blk.0: gate IQ4_XS, up IQ3_S; 31 of 48 layers differ), so the slot would read up rows with the wrong type. Such layers take the dense shared path (`qwen4_gemv` per tensor, each with its own type) at every row count.

- [ ] **Step 1: The failing check** — the reference is ds4's own CPU forward (`qwen4_ref_forward_token`, every weight through `qwen4_ref_row`, so each tensor with its own type). With the gateway paused (ask the user), run on the repacked model:

```bash
ISTA=~/orca/workspaces/ds4-metal-data/gguf/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-DS4-MTP.gguf
PLE=~/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
DS4_QWEN4_GPU=1 DS4_QWEN4_GPU_CHUNK=1 ./ds4 -m $ISTA --ple $PLE --metal -c 4096 --nothink \
  --first-token-test -p "Viết một câu ngắn về Hà Nội." 2>&1 | tail -3
```
Expected now: FAIL — the graph refuses the model (`unsupported dense weight type ...` / `needs ... dense weights`), because `qwen4_graph_dense_ok` and `qwen4_graph_expert_ok` do not know the types yet. Record the line.

Run the same command on Ivan's IQ2 model (`$D/ivan/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`) and record its `Qwen3.8 GPU-vs-CPU: ... worst max|diff|=X, top1 agree a/b` line: X is the normal GPU-vs-CPU gap for this architecture.

- [ ] **Step 2: Graph gates and dense routing**

```c
/* GSQ-RCO types in the Metal graph: qwen4_row_dot has them; CUDA does not. */
static bool qwen4_graph_gsq_ok(uint32_t type) {
#ifdef DS4_HAS_QWEN4_METAL
    return qwen4_type_is_gsq(type);
#else
    (void)type;
    return false;
#endif
}
```

- `qwen4_graph_dense_ok`: `... || t->type == DS4_TENSOR_Q4_K || qwen4_graph_gsq_ok(t->type));`
- `qwen4_graph_expert_ok`: add `|| (qwen4_graph_gsq_ok(t->type) && tensor_type(t->type) && (t->dim[0] % tensor_type(t->type)->block_elems) == 0)`.
- `qwen4_gemv_rows`, Apple `switch (w->type)`: the BF16 case takes the new types too:

```c
    case DS4_TENSOR_Q5_K: case DS4_TENSOR_Q6_K: case DS4_TENSOR_IQ2_XS: case DS4_TENSOR_IQ2_S:
    case DS4_TENSOR_IQ3_XXS: case DS4_TENSOR_IQ3_S: case DS4_TENSOR_IQ4_NL: case DS4_TENSOR_IQ4_XS:
    case DS4_TENSOR_Q2_0:
    case DS4_TENSOR_BF16: {
```
- the graph's "needs Q8_0/Q4_0/F16/BF16/F32 dense weights" message gains ", or a GSQ-RCO type".

Build (`make`), rerun Step 1's ISTA command with `DS4_QWEN4_GPU_CHUNK=1`.
Expected: the model loads and the check runs, but it FAILS on the shared expert: `top1 agree` below n/n and `worst max|diff|` far above Ivan's X (layers with different shared gate/up types read up rows with the gate's type). If it passes instead, the file has no such layer: stop and check `grep -c ffn_gate_shexp` against the allocation (31 layers differ) before going on.

- [ ] **Step 3: Shared gate/up types**

```c
    /* A decode batch runs the shared expert as dense projections over its
     * rows, as the prefill path does: as a slot of the per-token kernels it
     * is read once per row, 5 MB of Q8 per row and layer.  Single tokens and
     * verify rows keep the slot, unless the shared gate and up differ in type:
     * the slot kernels take one type for both (GSQ-RCO files mix them). */
#ifdef DS4_HAS_QWEN4_METAL
    const bool shared_dense = (T > 8u || l->ffn_gate_shexp->type != l->ffn_up_shexp->type) &&
        qwen4_graph_dense_ok(l->ffn_gate_shexp) && qwen4_graph_dense_ok(l->ffn_up_shexp) &&
        qwen4_graph_dense_ok(l->ffn_down_shexp);
```
(replacing the existing comment and `const bool shared_dense = T > 8u && ...` lines). The streamed-layer call below already passes `shared_dense ? UINT32_MAX : ...`, and the reduce already takes `sh_out` when `shared_dense`, so nothing else changes. PROD's shared gate and up are both Q8_0: its path is unchanged.

- [ ] **Step 4: Verify**

1. Step 1's ISTA command with `DS4_QWEN4_GPU_CHUNK=1`, `=8` and `=16` (slot path, verify-size rows, dense path): each prints `top1 agree n/n` and a `worst max|diff|` within 2× Ivan's X. Record the three lines.
2. `./ds4_test --metal-kernels` → `metal-kernels: OK`; CPU groups → `ds4 tests: ok`; `cd tools && python3 -m unittest discover -p 'test_*.py'` → OK.
3. `make cpu` → builds (the `#else` branch of `qwen4_graph_gsq_ok` compiles), then `make` again for the Metal binaries.
4. Restore the gateway.

- [ ] **Step 5: Commit**

```bash
git add ds4.c
git commit -m "qwen4: the Metal graph runs GSQ-RCO tensors; mixed shared gate/up types take the dense path"
```

---

### Task 9: End-to-end run, PROD checks, results (GPU: night, gateway paused)

**Files:**
- Create (not in git): `~/orca/workspaces/ds4-metal-data/sp3/run-e2e.sh`, `.../sp3/baseline/` (binaries + `metal/` from `feature/nextgen-qwen` before SP3)
- Create: `speed-bench/nextgen-eval/results/<date>-sp3-gsq-rco.md`
- Modify: `docs/QWEN38_FLASH_NEXT.md` (a GSQ-RCO paragraph)

- [ ] **Step 1: Baseline build** — copy `ds4`, `ds4-bench`, `metal/` from the kv-grow worktree (it builds `feature/nextgen-qwen` without SP3) into `~/orca/workspaces/ds4-metal-data/sp3/baseline/` after the SP4 batch ends.

- [ ] **Step 2: Write `~/orca/workspaces/ds4-metal-data/sp3/run-e2e.sh`** (not in git):

```zsh
#!/bin/zsh
# SP3 end to end (GPU, gateway paused): PROD identity + speed A/B, ISTA coherence, MTP, speed.
# Never kill -9 a Metal process: SIGTERM and wait.
R=/Users/dongnh/orca/workspaces/ds4-metal/sp3
S=/Users/dongnh/orca/workspaces/ds4-metal-data/sp3
B=$S/baseline
M=/Users/dongnh/.local/share/ai-gateway/ds4-models
PROD=$M/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf
PLE=$M/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
ISTA=/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-DS4-MTP.gguf
LA=/Users/dongnh/Library/LaunchAgents
PROD_ENV=(DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0 DS4_QWEN4_KV_GROW=1)
PROD_FLAGS=(--metal --ple $PLE --prefill-chunk 2048 --ssd-streaming --ssd-streaming-cache-experts 6GB)
BENCH=(--prompt-file $R/tests/long_context_story_prompt.txt --ctx-start 8192 --ctx-max 8192 --gen-tokens 128)
PROMPTS=("Write a Python function that checks whether a number is prime."
         "Giải thích ngắn gọn vì sao bầu trời có màu xanh."
         "List three differences between TCP and UDP.")
stamp() { date '+%H:%M:%S'; }

restore() {
  echo "== restore gateway $(stamp)"
  launchctl load $LA/dev.dongnh.ai-proxy.plist
  launchctl load $LA/dev.dongnh.gateway-watchdog.plist
  launchctl kickstart gui/$(id -u)/dev.dongnh.gateway-watchdog
}

while [ "$(curl -s --max-time 5 http://127.0.0.1:8090/status | python3 -c 'import json,sys
try: print(len(json.load(sys.stdin).get("active") or []))
except Exception: print(0)')" != "0" ]; do sleep 30; done
echo "== pause gateway $(stamp)"
launchctl unload $LA/dev.dongnh.gateway-watchdog.plist
launchctl unload $LA/dev.dongnh.ai-proxy.plist
trap restore EXIT
trap 'echo "== signal $(stamp)"; exit 1' INT TERM HUP
pid=$(cat /Users/dongnh/.local/share/ai-gateway/ds4-ornith.pid 2>/dev/null)
[ -n "$pid" ] && kill -TERM $pid 2>/dev/null
gw=$(lsof -nP -iTCP:8090 -sTCP:LISTEN -t 2>/dev/null)
[ -n "$gw" ] && kill -TERM $gw 2>/dev/null
for i in {1..60}; do pgrep -f ds4-server >/dev/null || break; sleep 5; done
pgrep -fl ds4-server && { echo "== ABORT: ds4-server still running"; exit 1; }

echo "== 1 PROD identity $(stamp)"
for i in 1 2 3; do
  for tag in baseline sp3; do
    dir=$B; [ $tag = sp3 ] && dir=$R
    (cd $dir && env $PROD_ENV ./ds4 -m $PROD $PROD_FLAGS -c 262144 --nothink --temp 0 -n 128 -p "${PROMPTS[$i]}") \
      > $S/id-$i-$tag.txt 2> $S/id-$i-$tag.log
  done
  cmp -s $S/id-$i-baseline.txt $S/id-$i-sp3.txt && echo "identity $i: identical" || echo "identity $i: DIFFERENT"
done

echo "== 2 PROD speed A/B $(stamp)"
for round in 1 2 3; do
  for tag in baseline sp3; do
    dir=$B; [ $tag = sp3 ] && dir=$R
    (cd $dir && env $PROD_ENV ./ds4-bench -m $PROD $PROD_FLAGS $BENCH --csv $S/bench-prod-$tag-$round.csv) \
      > $S/bench-prod-$tag-$round.log 2>&1
  done
done

echo "== 3 ISTA coherence + MTP $(stamp)"
for i in 1 2 3; do
  (cd $R && /usr/bin/time -p ./ds4 -m $ISTA --ple $PLE --metal -c 32768 --nothink --temp 0 -n 256 --mtp --mtp-timing \
     -p "${PROMPTS[$i]}") > $S/ista-$i.txt 2> $S/ista-$i.log
  grep "Qwen3.8 mtp:" $S/ista-$i.log
done

echo "== 4 ISTA speed $(stamp)"
for round in 1 2 3; do
  (cd $R && ./ds4-bench -m $ISTA --ple $PLE --metal $BENCH --csv $S/bench-ista-$round.csv) > $S/bench-ista-$round.log 2>&1
  (cd $R && DS4_QWEN4_MOE_MV_SPECIALIZE=1 ./ds4-bench -m $ISTA --ple $PLE --metal $BENCH \
     --csv $S/bench-ista-spec-$round.csv) > $S/bench-ista-spec-$round.log 2>&1
done
echo "== done $(stamp)"
```
(The baseline directory holds `ds4`, `ds4-bench` and `metal/`, so each binary loads its own shaders exe-relative. The `/usr/bin/time -p` lines in `ista-*.log` give the wall time including the Metal library compile.)

- [ ] **Step 3: Ask the user for the GPU window; run it; restore the gateway; read every output.**
Expected: identity PASS; PROD decode median sp3 ≥ 0.98 × baseline (else stop: the new `qwen4_row_dot` branches slowed the unspecialized multi-GEMV; fix by moving the GSQ cases into a separate multi-GEMV kernel, test first); ISTA answers coherent in all 3 prompts (read them); MTP acceptance and speeds recorded.

- [ ] **Step 4: Results and docs** — write the results file (inputs with sha256, repack summary, identity, PROD A/B table, ISTA table, MTP acceptance, the reconciliations 3, 4 and 9 as open items for sub-project 5) and a `docs/QWEN38_FLASH_NEXT.md` paragraph: which GSQ-RCO types ds4 runs, how to repack, what is not supported (tiled prefill, SSD-streamed layers, CUDA).

- [ ] **Step 5: Commit**

```bash
git add speed-bench/nextgen-eval/results/*-sp3-gsq-rco.md docs/QWEN38_FLASH_NEXT.md
git commit -m "sp3 results: ISTA GSQ-RCO IQ3_XXS runs on ds4"
```

- [ ] **Step 6: HF** — ask the user before uploading the repacked GGUF and manifest to dongnhdev.
