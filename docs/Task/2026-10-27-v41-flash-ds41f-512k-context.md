# ds41f (DeepSeek-V4.1-Flash-1017) — cấu hình 512k context và tok/s theo ngữ cảnh

Ngày: 2026-10-27. Máy: M5 Pro 64 GB. Model: `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-ds41.gguf`
(405,224,791,632 B, 1010 tensor / 130 KV). Binary: `ds4` (fork ds4-metal).

## 1. Metadata liên quan tới context

| Khoá | Giá trị |
|---|---|
| `context_length` | **1,048,576** (1M) |
| `block_count` | 40 |
| `attention.head_count` / `head_count_kv` | 64 / **1** |
| `attention.key_length` / `value_length` | 512 / 512 |
| `attention.kv_source_layer_ids` | **[2, 8, 14, 20]** (chỉ 4 layer giữ KV) |
| `attention.compress_ratios` | layer 0–1: 1, layer 2–19: 2, layer 20+: 1 |
| `attention.indexer.top_k` | 512 (DSA sparse attention) |
| `rope.scaling` | yarn, factor 16, original 65536 |

⇒ 512k **nằm trong dải hỗ trợ** của model và rẻ về KV: đo thực tế **KV 3.13 GiB** cho ctx 524288.

## 2. Cấu hình chạy 512k

```bash
bash ~/.local/bin/server-power.sh off        # ds4 giữ lock /tmp/ds4.lock
scripts/ds41f-ctx.sh -c 524288 --prompt-file /tmp/p128k.txt -n 64
bash ~/.local/bin/server-power.sh on         # dựng lại prod 35B
```

Memory plan khi khởi động (fork tự co cache):

```
memory: KV 3.13 GiB (compressed 3.13) + buffers 10.08 GiB + resident model 7.45 GiB
        + expert cache 26.21 GiB + prefill expert reserve 7.12 GiB = 53.99 GiB planned
V4.1 SSD cache fitted from 4300 to 3595 experts for context/runtime headroom
→ dynamic cache 2827 experts (9.49 MiB/expert)
```

Đánh đổi: ctx 512k ⇒ context buffers **13.22 GiB** ⇒ cache expert 4900 → **2827** ⇒ decode ngắn
giảm từ 15.65 t/s xuống **12.85 t/s** (đo với prompt ~30 token).

## 3. Đo tok/s theo ngữ cảnh (prefill + decode)

Prompt sinh tổng hợp (tiếng Anh, lặp đoạn văn), `-n 64`, `--temp 0`, cache 4300, ctx 524288.

| Prompt tokens | Prefill t/s | Decode t/s | Wall |
|---|---|---|---|
| 1,978 | 114.5 | 8.60 | 25.6 s |
| 7,522 | 275.0 | 8.47 | 35.7 s |
| 30,118 | 333.4 | 7.37 | 100.4 s |
| 60,190 | 401.2 | 7.01 | 160.6 s |
| 131k | (đang đo) | | |
| 262k | (đang đo) | | |
| 524k | (đang đo) | | |

Ghi chú: prefill tăng dần theo độ dài prompt (chi phí cố định được amortize) — 512k prefill
ước ~20–25 phút ở ~400 t/s.

## 4. Vì sao decode giảm theo context

Ba yếu tố cộng dồn:
1. **Context buffers + cache co lại** (4900 → 2827 expert) ⇒ nhiều expert miss hơn ⇒ refill
   ~6.6 MB/miss (xem báo cáo tuning §12), đây là phần lớn mức giảm 15.65 → 12.85 t/s.
2. **Attention dài**: DSA chọn top-512 block/token, chi phí indexer + attention tăng theo log/độ dài.
3. **KV đọc thêm** 3.13 GiB compressed (mỗi token chỉ đọc phần được chọn).
