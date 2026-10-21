#!/usr/bin/env python3
"""Header audit for Ornith-1.5-122B tiers: which quant each tensor family uses.

The engine accepts a fixed set of row geometries per tensor family
(tensor_expect_qwen35_*_layout + qwen4_expert_row_bytes); this reports what a
tier file actually ships so the gate is widened against evidence, not hope."""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from gguf_lite import Reader, BLOCK  # noqa: E402

# engine-accepted routed-expert types (qwen4_expert_row_bytes ds4_metal.m) by gguf id
ROUTED_OK = {8, 10, 12, 13, 14, 16, 39}
DENSE_OK = {0, 1, 2, 6, 8, 10, 11, 12, 13, 14, 30, 39}
NAME = {8: 'Q8_0', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K',
        16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 20: 'IQ4_NL', 21: 'IQ3_S',
        22: 'IQ2_S', 23: 'IQ4_XS', 30: 'BF16', 39: 'MXFP4', 42: 'Q2_0',
        0: 'F32', 1: 'F16', 2: 'Q4_0', 6: 'Q5_0'}

FAMILY = defaultdict(set)


def fam(name):
    n = name.lower()
    if 'ffn_gate_exps' in n:
        return 'gate_exps'
    if 'ffn_up_exps' in n:
        return 'up_exps'
    if 'ffn_down_exps' in n:
        return 'down_exps'
    if 'ffn_gate_shexp' in n:
        return 'gate_shexp'
    if 'ffn_up_shexp' in n:
        return 'up_shexp'
    if 'ffn_down_shexp' in n:
        return 'down_shexp'
    if 'ffn_gate_inp' in n:
        return 'router'
    if 'attn_qkv' in n or 'linear_attn' in n:
        return 'attn_qkv(GDN)'
    if 'attn_q' in n:
        return 'attn_q'
    if 'attn_k' in n:
        return 'attn_k'
    if 'attn_v' in n:
        return 'attn_v'
    if 'attn_output' in n:
        return 'attn_out'
    if 'nextn' in n or n.startswith(('blk.48.', 'block.48.')):
        return 'mtp_head'
    if 'output' in n:
        return 'output_head'
    if 'token_embd' in n:
        return 'token_embd'
    return 'other'


def main(path):
    r = Reader(path)
    print(f'== {Path(path).name} ==')
    m = {k: v[1] for k, v in r.kv.items()}
    arch = m.get('general.architecture', '?')
    print(f'arch={arch} tensors={len(r.tensors)}',
          'quantize="' + str(m.get('general.quantization', '')) + '"',
          'size_label="' + str(m.get('general.size_label', '')) + '"')
    total = 0
    for t in r.tensors:
        name, ty, dims = t['name'], t['type'], t.get('dims') or []
        el, by = BLOCK.get(ty, (0, 0))
        total += t['nbytes']
        FAMILY[fam(name)].add(ty)
    print(f'total tensor bytes: {total:,}')
    for f in sorted(FAMILY):
        types = FAMILY[f]
        bad = sorted(ty for ty in types if not (ty in ROUTED_OK if 'exps' in f and 'shexp' not in f and 'rider' not in f and 'inp' not in f else ty in DENSE_OK))
        flag = '' if not bad or f == 'other' else '  <-- REFUSED BY THE WIDE GATE'
        print(f'{f:14s} ' + ', '.join(NAME.get(ty, f'type{ty}') for ty in sorted(types)) + flag)


if __name__ == '__main__':
    main(sys.argv[1])
