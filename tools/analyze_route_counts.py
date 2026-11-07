#!/usr/bin/env python3
"""Analyze DS4_QWEN4_ROUTE_COUNTS output: per-layer expert popularity.

Usage: python3 analyze_route_counts.py /tmp/qwen38-route-counts.txt
"""
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen38-route-counts.txt"
counts = defaultdict(lambda: defaultdict(int))
total = 0
layers = set()
experts = set()
for line in open(path):
    line = line.strip()
    if not line or line.startswith("#"):
        if "total_selections" in line:
            total = int(line.split()[-1])
        continue
    l, e, h = map(int, line.split())
    counts[l][e] += h
    layers.add(l)
    experts.add(e)
    total += h

print(f"layers={len(layers)} experts_seen={len(experts)} total_selections={total}")
KS = [8, 16, 32, 64, 128, 256]
agg = {k: 0 for k in KS}
per_layer = []
for l in sorted(counts):
    vals = sorted(counts[l].values(), reverse=True)
    s = sum(vals)
    row = []
    for k in KS:
        cov = sum(vals[:k]) / s if s else 0
        agg[k] += sum(vals[:k])
        row.append(cov)
    per_layer.append((l, s, row))
    print(f"layer {l:3d} sel={s:8d} " + " ".join(f"top{k}={c:.3f}" for k, c in zip(KS, row)))

grand = sum(sum(c.values()) for c in counts.values())
print("\naggregate coverage of all selections:")
for k in KS:
    print(f"  top-{k:3d} experts: {agg[k]/grand:.4f}")

# expert reuse across layers: how concentrated is the global hot set?
flat = defaultdict(int)
for l in counts:
    for e, h in counts[l].items():
        flat[e] += h
vals = sorted(flat.values(), reverse=True)
gs = sum(vals)
print("\nglobal expert frequency (summed over layers):")
for k in KS:
    print(f"  top-{k:3d}: {sum(vals[:k])/gs:.4f}")
print(f"  distinct experts used: {len(flat)} / {max(experts)+1}")
