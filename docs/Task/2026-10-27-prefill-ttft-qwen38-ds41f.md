# Prefill & TTFT: Ornith-35B (Qwen3.8-arch, prod) vs ds41f-1017 — đo 2026-10-27

Máy: M5 Pro 64 GB. Prod: `ds4-server` trên :18087 sau gateway :8090.

## 1. Ornith-1.5-35B-A3B (model prod, fully resident 22.4 GiB)

Lệnh prod: `ds4-server --metal -m Ornith-...-23G-ICE.gguf -c 262144 --mtp --kv-disk-dir ds4-kv-cache-ornith
--kv-disk-space-mb 32768 --kv-cache-cold-max-tokens 262144 --kv-cache-min-tokens 4096
--rewind-point-min-tokens 16384 --vision mmproj-Ornith...`

| Đo | Kết quả |
|---|---|
| CLI `-c 16384` (8k-word prompt) | prefill **2159 t/s**, decode **70.8 t/s** |
| CLI `-c 262144 --mtp` | prefill **2057 t/s**, decode 74.2 (ctx lớn + MTP chỉ −2.5%) |
| CLI `-c 16384` (2k-word) | prefill **2112 t/s**, decode 70.4 |
| Server, prompt **53,236 token** | 36.6 s (không KV-disk) / 39.0 s (prod có KV-disk) ⇒ **~1400 t/s** |
| KV-disk | **không phải nguyên nhân** (36.6 vs 39.0 s) |
| Gateway overhead | prompt "Say OK." (283 token): TTFT gateway **0.26 s** vs backend trực tiếp 0.68 s ⇒ gateway không cộng đáng kể |
| TTFT thực tế | ≈ 0.3 s + tokens/1400 |

⇒ Prod **không có lỗi cấu hình**: TTFT dài là do prefill tuyến tính (~0.7 ms/token). Prompt 50k token ⇒ ~36 s.

## 2. ds41f-1017 (DeepSeek-V4.1-Flash, streaming 377 GiB, cache 4300→2827, ctx 524288)

| Prompt tokens | Prefill t/s | TTFT | Decode t/s |
|---|---|---|---|
| 1,978 | 114 | 17.3 s | 8.60 |
| 7,522 | 275 | 27.3 s | 8.47 |
| 30,118 | 333 | 90.4 s | 7.37 |
| 60,190 | 401 | 150 s | 7.01 |
| 120,334 | 436 | 276 s | 6.19 |

- TTFT ~17 s ngay cả với prompt 2k ⇒ có **~12–15 s khởi động/warmup** (page-fault expert weights + seed cache);
  sau đó prefill ~2.3 ms/token. Đây là lý do cảm giác "rất chậm" với prompt ngắn.
- `DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR=1`: không cải thiện (108 vs 114 t/s @2k; 264 vs 275 @8k).

## 3. Qwen3.8-Flash-Next (bản 41G) — **chạy bình thường, cần `--ple` sidecar**

Lỗi "lacks its n-grams" ở lần đo đầu là do **thiếu flag `--ple`**: model này cần sidecar
`Qwen3.8-Flash-Next-PLE-Q4_1.gguf` (30 GB) chứa PLE/n-gram. Lệnh đúng (theo
`~/.local/ai-gateway/runtime-registry.json`):

```
env DS4_QWEN4_STREAM_FULL_LAYERS=32 DS4_QWEN4_PLE_PREFETCH_FULL=0 \
    DS4_QWEN4_MTP_DRAFT_VOCAB=.../Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt \
    DS4_QWEN4_KV_GROW=1 \
  ds4-server --metal -m Qwen3.8-...-Q2KDownPad768-DenseQ4Kselimat-MTP.gguf \
    --ple Qwen3.8-Flash-Next-PLE-Q4_1.gguf -c 262144 --prefill-chunk 2048 --mtp \
    --kv-disk-dir .../ds4-kv-cache --kv-disk-space-mb 32768 --kv-cache-cold-max-tokens 262144 \
    --think-budget 4096 --host 127.0.0.1 --port 18086 --vision mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
```

Đo thực tế (M5 Pro 64 GB, một mình model này):

| Đo | Kết quả |
|---|---|
| Memory plan | KV 6.72 GiB (raw 6.50 + compressed 0.22) + buffers 3.06 + **resident 40.67 = 50.45 GiB** |
| Thời gian ready | **3–6 s** |
| Prefill, `--prefill-chunk 2048` | 32,487 token → 45.6 s = **713 t/s** |
| Prefill, `--prefill-chunk 8192` | 40,862 token → 51.0 s = **802 t/s** (**+12%**) |
| Decode | **46.8 t/s** |
| TTFT cold (không prefix cache) | ≈ tokens/700–800 s (2.7k → 11.7 s; 32k → 45.6 s) |
| TTFT warm (trùng prefix) | **3.3–4.1 s** (chỉ prefill phần mới, ~510 t/s) |

So sánh: Ornith-35B prod nhanh hơn ~2–3× (1400–2150 t/s prefill, 70–74 t/s decode) vì model
nhỏ hơn và fully resident 22.4 GiB, trong khi Qwen3.8 là 136B/512 expert ở 40.67 GiB resident
+ layer streaming ⇒ prefill bị chặn bởi đọc expert.

**Ghi chú quan trọng**
- `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` là **sidecar bắt buộc** — KHÔNG được xoá (30 GB).
- Bản 48G `...Q4KDownTrim...` chỉ là bản trùng: bản trong `ds4-models/` đã xoá, nhưng vẫn còn
  một bản ở `~/orca/workspaces/ds4-metal-data/gguf/` (không mất dữ liệu).
- Bản `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` không load độc lập được (nó là sidecar, không phải model chính).

## 4. Kết luận & việc nên làm

| Model | Prefill | TTFT (prompt 10k) | Trạng thái |
|---|---|---|---|
| Ornith-35B (prod) | ~1400–2150 t/s | ~7 s | tốt nhất có thể, không lỗi cấu hình |
| ds41f-1017 | 114–436 t/s | ~25 s (+17 s warmup) | bản chất model streaming 377 GB |
| Qwen3.8-Flash 41G (+`--ple`) | 713–802 t/s | 45.6 s (32k) / 4 s warm | chạy tốt; prefill ~1/2–1/3 Ornith |

Việc nên làm:
1. **Qwen3.8 41G**: dùng `--prefill-chunk 8192` (đo được +12% prefill); giữ cả model 41G **và** sidecar PLE-Q4_1.
2. **ds41f**: nếu dùng tương tác, chạy 1 warmup request lúc khởi động để trả 12–15 s warmup trước.
3. **Prod**: giữ nguyên; nếu prompt thường >30k token và cần TTFT thấp hơn, có thể thử `--prefill-chunk 8192/16384`
   (chưa đo vì cần dừng prod).
