
## Addendum: untouched MoE components (reduce/router/shared)

kernel_qwen4_moe_reduce reads `part` = [T][S+1][dim] fp32 written by the
down kernel: ~208 MB per layer, ~10 GB per prefill chunk of pure
write+read traffic (~25-50 ms of the measured 79 ms reduce stage is
DRAM-bound floor). The kernel itself is already coalesced; the remaining
cost is the intermediate buffer. Candidate (unverified): store `part` as
fp16 + error compensation like the midh/midr path, or fuse the weighted
reduce into the down epilogue - both touch logits accuracy and need the
full correctness suite. Router (76 ms) and shared expert (68 ms) not yet
profiled at kernel level.
