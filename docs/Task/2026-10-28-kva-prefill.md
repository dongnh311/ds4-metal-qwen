# KVA prefill cho Qwen3.8 — kết quả (2026-10-28)

## Phase 0 — parity nguồn

Binary prod cũ (12/10) chứa các knob tuning không có trong cây nguồn nào trên máy
(`DS4_QWEN4_GDN_PREFILL_NSG`, `MOE_MM_NAX`, `ATTN_MERGE_WIDE`…). Đã đo 4 binary
có sẵn, cùng model/prompt, bằng `scripts/phase0-parity.py`:

| binary | prefill 24k | decode |
|---|---|---|
| prod 12/10 | 745,3 t/s | 50,1 |
| ornith-122b (27/10) | 741,8 (−0,5%) | 50,0 |
| develop (10/10) | 745,1 (−0,0%) | 49,7 |
| foxface (3/10) | 739,2 (−0,8%) | 50,0 |

⇒ Các knob đó chỉ là nhánh tuning thay thế; build lại từ `ornith-122b` không
chậm đi. Chốt base = `ornith-122b`.

## Implement

`ds4.c`:
- `qwen4_graph_kva_begin(g, n_tokens)`: đặt `g->kva_until = n_tokens - tail`
  (gọi 1 lần trước vòng chunk, cả đường CLI lẫn server).
- Chunk có `pos0 + T <= kva_until`: layer `il >= split` chạy **state-only**.
  - attention: qkv + rope + ghi KV + indexer block keys, dừng trước attention core
    và out-proj (`qwen4_graph_attention_tail(..., state_only)`).
  - GDN: front + prep + scan (cập nhật recurrence), dừng trước `gdn_out`.
  - bỏ `hc_combine`, `hc_ffn_mix`, MoE ⇒ `g->R` giữ hidden của layer `split-1`.
- Chunk phủ tail chạy đầy đủ ⇒ decode khởi động từ state chính xác.
- Env: `DS4_QWEN4_KVA_SPLIT` (0 = tắt), `DS4_QWEN4_KVA_TAIL` (mặc định 2048,
  clamp 512..8192). Prompt < 1,5×tail không xấp xỉ ⇒ agent prompt ngắn không đổi.

## Đo (chunk 4096, ctx 262144, tail 2048)

| điểm đo | off | split 36 | Δ |
|---|---|---|---|
| 24k, prefill | 723–746 t/s | 871 t/s | **+17%** |
| 24k, split 32 / 40 / 44 | — | 902 / 802 / 767 t/s | +25% / +11% / +6% |
| 52k, prefill (cold) | 74 s (718 t/s) | **58,3 s (897 t/s)** | **+27%** |
| 128k, prefill (cold) | 181 s (719 t/s) | **150,9 s (863 t/s)** | **+20%** |
| decode sau prompt | 50,1 t/s | 50,1 t/s | 0% |
| memory | 48,83 GiB | 48,83 GiB | 0 |

## Cổng chất lượng

| test | split 36 | split 32 |
|---|---|---|
| needle 52k (3 mã ở 15/50/85%) | **3/3**, 58 s | 3/3, 57 s |
| needle 128k | **3/3**, 151 s | chưa đo |

## Quyết định

Bật `DS4_QWEN4_KVA_SPLIT=36`, `DS4_QWEN4_KVA_TAIL=2048` cho Qwen3.8; split 32
nhanh hơn ~7% ở 24k nhưng chưa có bằng chứng chất lượng ngoài needle ⇒ để dành,
cần bộ extraction đầy đủ trước khi dùng.

Rollback: bỏ 2 env, restart backend (không cần đổi code). Khi tắt, đường chạy
byte-identical với build trước (đã đo 745,9 vs 745,3 t/s).

## P2 — ds41f lookahead: kết quả audit

`ds41_la_post_layer` chỉ được gọi từ nhánh **decode** (MoE một token,
`ds4.c` ≈43695); đường prefill batch (T>1) **không** post job nào. Job hiện tại
mang 1 vector `x` + `pos` nên không dùng lại trực tiếp cho batch được — muốn
prefetch cho prefill phải đọc selected ids của cả chunk và warm theo union.

Vì vậy: sweep `DS4_METAL_V41_LOOKAHEAD_K` hiện chỉ tác động decode. Cần cửa sổ
dừng Qwen ~30–40 phút để nạp ds41f và đo k=0/1/3/6 (prefill 8k/30k, decode
4k/30k). Chưa làm để không hạ prod khi chưa có số.

## Tái lập

```
scripts/phase0-parity.py off=./ds4-server "kva=./ds4-server|DS4_QWEN4_KVA_SPLIT=36,DS4_QWEN4_KVA_TAIL=2048"
scripts/needle.py --tag kva36-52k --binary $PWD/ds4-server --tokens 52000 --ctx 131072 \
                  --env DS4_QWEN4_KVA_SPLIT=36,DS4_QWEN4_KVA_TAIL=2048
```