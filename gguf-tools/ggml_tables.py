#!/usr/bin/env python3
"""GGML dequant support tables, parsed from the upstream ggml-common.h snapshot (fetched from
ggml-org/llama.cpp master on 2026-10-21 and cached at tools/ggml-common.h if present). The arrays
are serialized to npz next to this file on first build so later runs never depend on the header."""
import re
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HDRS = [HERE / "ggml-common.h", Path("/tmp/ggml-common.h")]
NPZ = HERE / "ggml_tables.npz"


def _parse(path):
    src = path.read_text()
    want = {"iq2xxs_grid": ("u64", 256), "iq2xs_grid": ("u64", 512), "iq2s_grid": ("u64", 1024),
            "iq3xxs_grid": ("u32", 256), "ksigns_iq2xs": ("u8", 128), "kmask_iq2xs": ("u8", 8)}
    out = {}
    for name, (kind, count) in want.items():
        m = re.search(r"GGML_TABLE_BEGIN\(uint(?:8|16|32|64)_t, %s, (\d+)\)(.*?)GGML_TABLE_END\(\)" % name,
                      src, re.S)
        if not m:
            raise ValueError(f"table {name} not found in {path}")
        body = m.group(2)
        if kind == "u64":
            vals = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]{6,16}", body)]
            arr = np.array(vals, dtype=np.uint64)
        elif kind == "u32":
            vals = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]{6,10}", body)]
            arr = np.array(vals, dtype=np.uint32)
        else:
            vals = [int(x) for x in re.findall(r"\d+", body)]
            arr = np.array(vals, dtype=np.uint8)
        if len(arr) != m.group(1) and len(arr) != count:
            raise ValueError(f"table {name}: parsed {len(arr)} entries, want {count}")
        if len(arr) != count:
            raise ValueError(f"table {name}: parsed {len(arr)}, want {count}")
        out[name] = arr
    return out


def load():
    if NPZ.exists():
        z = np.load(NPZ)
        return {k: z[k] for k in z.files}
    for h in HDRS:
        if h.exists():
            tables = _parse(h)
            np.savez(NPZ, **tables)
            return tables
    raise FileNotFoundError("no ggml-common.h snapshot available")


def u64_bytes(arr, idx):
    """The 8 bytes of grid entry idx, LSB first (the (const uint8_t*)(grid+i) view)."""
    return struct.pack("<Q", int(arr[idx]))
