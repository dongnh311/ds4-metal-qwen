#!/usr/bin/env python3
"""Generate the Q5_K tiled-GEMM kernels for metal/qwen35.metal from the Q4_K
templates in metal/qwen4.metal by an exact slice + literal replacement.

Deterministic: it fails loudly if an anchor is missing or matches more than
once, so a qwen4.metal edit can never silently produce the wrong kernel.

The tensor-op (NAX) templates and their instantiations are wrapped in
#ifdef DS4_METAL_HAS_TENSOR: without it, DS4_METAL_DISABLE_METAL4=1 builds
and pre-M5 devices fail to compile the whole Metal library (identifiers such
as matmul2d_descriptor only exist under the tensor API), which would also
break the Qwen3.8 path since both families share one .metallib.

Re-running this script is byte-stable: it never grows a blank line between
the generated block and whatever follows it in metal/qwen35.metal.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "metal", "qwen4.metal")
DST = os.path.join(ROOT, "metal", "qwen35.metal")
MARK = "// BEGIN GENERATED q5_K tiles (extract_q5k_tiles.py) — do not edit by hand"
END = "// END GENERATED q5_K tiles"


def slice_between(text, start_anchor, end_anchor):
    i = text.find(start_anchor)
    j = text.find(end_anchor)
    if i < 0 or j < 0 or text.count(start_anchor) != 1 or text.count(end_anchor) != 1 or j <= i:
        sys.exit("extract: anchors not found uniquely: %r .. %r" % (start_anchor, end_anchor))
    return text[i:j]


def main():
    src = open(SRC, encoding="utf-8").read()
    # mid template + its four instantiations: from the mid doc-comment to the
    # down doc-comment.
    mid = slice_between(src,
        "/* mid[t][slot][r] = silu(gate . x) * (up . x) for every (token, slot) routed",
        "/* part[t][slot][r] = down . mid[t][slot], same tiling with mid as B */")
    # down template + its four instantiations: from the down doc-comment to the
    # rows_f32_to_f16 doc-comment.
    down = slice_between(src,
        "/* part[t][slot][r] = down . mid[t][slot], same tiling with mid as B */",
        "/* rows of floats -> halves (round to nearest even), four values per thread;")
    # the two tensor-op templates (mid then down) live between the header comment
    # and the instantiation macros; take from the mid nax kernel signature to the
    # QWEN4_NAX_MID_SIG_HALF macro (which starts the instantiations).
    naxmid = slice_between(src,
        "template <int NR1, typename XT, bool COMP>\nkernel void kernel_qwen4_moe_mm_mid_nax_t(",
        "template <int NR1, typename XT, bool COMP>\nkernel void kernel_qwen4_moe_mm_down_nax_t(")
    naxdown = slice_between(src,
        "template <int NR1, typename XT, bool COMP>\nkernel void kernel_qwen4_moe_mm_down_nax_t(",
        "#define QWEN4_NAX_MID_SIG_HALF")

    repl = [
        ("kernel_qwen4_moe_mm_mid_nax_t", "kernel_qwen35_moe_mm_mid_q5k_nax_t"),
        ("kernel_qwen4_moe_mm_down_nax_t", "kernel_qwen35_moe_mm_down_q5k_nax_t"),
        ("kernel_qwen4_moe_mm_mid", "kernel_qwen35_moe_mm_mid_q5k"),
        ("kernel_qwen4_moe_mm_down", "kernel_qwen35_moe_mm_down_q5k"),
        ("qwen4_mm_stage16", "qwen35_mm_stage16_q5k"),
        ("qwen4_load_raw16", "qwen35_load_raw_q5k"),
        ("qwen4_dequant_raw16", "qwen35_dequant_raw_q5k"),
        ("qwen4_raw16", "qwen35_raw_q5k"),
    ]

    def apply(block):
        for a, b in repl:
            block = block.replace(a, b)
        return block

    # The mid/down simdgroup slices already END WITH qwen4's four
    # `template [[host_name("kernel_qwen4_moe_mm_{mid,down}{,_nt1,_nt2,_nt8}")]]`
    # instantiation lines, which the replacement table renames to the q5k
    # names — so those 8 instantiations already exist in the mid/down blocks.
    # Do NOT emit them again (a second `template [[host_name(...)]]` for the
    # same specialization is a "duplicate explicit instantiation" and the
    # Metal library fails to compile).  Only the four NAX instantiations are
    # missing: the naxmid/naxdown slices stop at the QWEN4_NAX_*_SIG_* macros,
    # before the tensor-op instantiation lines, and this plan needs only the
    # half / non-compensated variants.
    #
    # The NAX templates and their instantiations only compile under the
    # tensor API (Fix 1): wrap them in one #ifdef DS4_METAL_HAS_TENSOR that
    # covers both, closed by the single #endif at the end of the appended
    # instantiation block.
    body = apply(mid) + apply(down) + "#ifdef DS4_METAL_HAS_TENSOR\n" + apply(naxmid) + apply(naxdown)
    body += """
template [[host_name("kernel_qwen35_moe_mm_mid_q5k_nax")]]   kernel void kernel_qwen35_moe_mm_mid_q5k_nax_t<32, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device half *, device half *, threadgroup char *, uint3, ushort, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mm_mid_q5k_nax64")]] kernel void kernel_qwen35_moe_mm_mid_q5k_nax_t<64, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device half *, device half *, threadgroup char *, uint3, ushort, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mm_down_q5k_nax")]]   kernel void kernel_qwen35_moe_mm_down_q5k_nax_t<32, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device const half *, threadgroup char *, uint3, ushort, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mm_down_q5k_nax64")]] kernel void kernel_qwen35_moe_mm_down_q5k_nax_t<64, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device const half *, threadgroup char *, uint3, ushort, ushort, ushort);
#endif
"""

    text = open(DST, encoding="utf-8").read()
    if MARK in text:
        head = text[:text.index(MARK)].rstrip("\n")
        tail_after = text[text.index(END) + len(END):].lstrip("\n")
    else:
        head = text.rstrip("\n")
        tail_after = ""

    new_block = MARK + "\n" + body.strip("\n") + "\n" + END

    parts = [head, new_block]
    if tail_after:
        parts.append(tail_after)
    text = "\n\n".join(parts) + "\n"

    open(DST, "w", encoding="utf-8").write(text)
    print("extract: wrote", DST)


if __name__ == "__main__":
    main()
