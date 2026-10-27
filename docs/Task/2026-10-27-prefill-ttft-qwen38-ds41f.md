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

## 3. Qwen3.8-Flash-Next (bản 41G bạn quan tâm) — **hiện KHÔNG load được**

```
$ ds4 -m Qwen3.8-...Q2KDownPad768-DenseQ4Kselimat-MTP.gguf ...
ds4: Qwen GGUF lacks its n-grams; repack with gguf-tools/qwen4_native_ngrams.py
```
- Cả binary repo (hôm nay) và binary prod (12/10) đều từ chối ⇒ file thiếu tensor/metadata n-grams.
- Bản 48G (`Q4KDownTrim`) lỗi y hệt; **đã xoá theo yêu cầu "giữ 1 bản 41G"** (giải phóng 48 GB).
- Bản `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` (30G) là format khác: `required metadata key is missing: deepseek4.block_count`.
- Registry gateway có 2 entry Qwen3.8 nhưng chúng trỏ tới **runtime remote 127.0.0.1:18086** (không trả lời),
  không phải ds4 local ⇒ hiện prod không phục vụ Qwen3.8.
- Muốn dùng/đo bản 41G: cần repack lại n-grams:
  ```
  python3 gguf-tools/qwen4_native_ngrams.py \
      --model  Qwen3.8-...Q2KDownPad768-DenseQ4Kselimat-MTP.gguf \
      --source <HF snapshot hoặc native n-gram GGUF> \
      --output Qwen3.8-...-ngrams.gguf --source-revision <rev>
  ```
  Checkpoint nguồn **không có** trong `~/.cache/huggingface` (chỉ có drafter/refusal-projection) ⇒ phải tải.

## 4. Kết luận & việc nên làm

| Model | Prefill | TTFT (prompt 10k) | Trạng thái |
|---|---|---|---|
| Ornith-35B (prod) | ~1400–2150 t/s | ~7 s | tốt nhất có thể, không lỗi cấu hình |
| ds41f-1017 | 114–436 t/s | ~25 s (+17 s warmup) | bản chất model streaming 377 GB |
| Qwen3.8-Flash 41G | — | — | **hỏng n-grams, cần repack** |

Việc nên làm:
1. **Qwen3.8 41G**: tải checkpoint nguồn và repack n-grams (tôi làm được, cần bạn đồng ý tải ~vài chục GB).
2. **ds41f**: nếu dùng tương tác, chạy 1 warmup request lúc khởi động để trả 12–15 s warmup trước.
3. **Prod**: giữ nguyên; nếu prompt thường >30k token và cần TTFT thấp hơn, có thể thử `--prefill-chunk 8192/16384`
   (chưa đo vì cần dừng prod).
