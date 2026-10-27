# DeepSeek-V4.1-Flash-1017 (ISTA GSQ-RCO IQ2_XXS) — bring-up trên ds4 + tối ưu tốc độ

Ngày: 2026-10-26 (đêm) · Máy: Apple M5 Pro, 64 GB unified, ~1 TB SSD · Fork: `feature/ornith-122b`

## 1. Kết quả cuối (tóm tắt cho buổi sáng)

| Hạng mục | Kết quả |
|---|---|
| Model chạy được trên ds4 | ✅ **16.12–16.13 t/s** decode (ctx 2048, greedy, prompt chuẩn TCP) |
| So với lúc bắt đầu | 12.68 t/s → **+27%** |
| So với default shipped của fork (12.1 t/s @8K) | **+33%** |
| File model | `~/.local/share/ai-gateway/ds4-models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-ds41.gguf` — **405,224,791,632 B (377.4 GiB)** |
| Prod 35B | ✅ đã dựng lại (`server-power.sh on` → backend_ok, gateway :8090 trả lời OK, KV-disk nguyên vẹn) |
| Disk | ~45 GiB free |
| 20 t/s | ❌ chưa đạt — cần việc kernel-level (mục 6) |

## 2. Chuỗi build đã hoàn tất (data-side, `ds4.c` không đổi)

1. **Merge** 4 shard ISTA + companion Engram thành 1 GGUF (226.6 GiB) — `deepseek41_ista_merge.py`.
2. **Grow KV**: chèn 50 KV ds4 yêu cầu (flat `deepseek41.*`, engram `token_map` 517 KB / `primes` / `multipliers` / `pad_id` / `rows`, các array i32→u32) + shift data 16 KiB-aligned — `deepseek41_ista_grow.py`.
3. **Engram retype**: `engram_wkv` (Q5_K) → `engram_kv` (F16), `engram_q/k` (BF16) → `engram_q_norm/k_norm` (F32), đặt trước bảng Engram (page-aligned) — `deepseek41_ista_engram_fix.py`.
4. **Requant toàn bộ** package sang dtype ds4 chấp nhận — `deepseek41_ista_retype.py` + `ds41_retype.c`:
   - dense Q5_K → Q8_0/Q4_K, `token_embd` → F16, `hc_*_fn` F32 → F16, `ffn_gate_inp` BF16 → F32, compressor/indexer BF16 → F16;
   - routed experts IQ1_M/IQ2_XS → IQ2_XXS;
   - `attn_output_a` 3-D → 2-D (4096, 8192).
5. **Smoke** `./ds4 --metal --ssd-streaming`: prefill ~10 t/s, decode 12.85→16.1 t/s; trả lời tiếng Việt đúng (khoa học, toán).

## 3. Tối ưu tốc độ (đo bằng prompt chuẩn, `-n 512`, temp 0, greedy)

| Bước | Thay đổi | t/s |
|---|---|---|
| Ban đầu | auto cache, resident Q8_0 | 12.7 |
| 1 | `--ssd-streaming-cache-experts 4200` (38.9 GiB cache, bỏ prefill reserve) | 13.6–13.9 |
| 2 | dense Q8_0 → **Q4_K** (attn_q_a/q_b/kv, shexp, output): resident 9.37 → 7.45 GiB | 15.5–16.2 |
| 3 | `ffn_down_exps` Q2_K → **IQ2_XXS** in-place (nguồn native, header 10→16) | +0.2 |
| 4 | `--ssd-streaming-cache-experts 4900` (41.6 GiB, trần mlock) | **16.12–16.13** |
| 5 | cache 5200 | 14.9 (bị nhả margin mlock — kém hơn) |

**Cấu hình tốt nhất:**
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/ornith-122b
./ds4 -m ~/.local/share/ai-gateway/ds4-models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-ds41.gguf \
  --metal --ssd-streaming -c 2048 --ssd-streaming-cache-experts 4900 \
  -p "..." -n 512 --temp 0
```
Các env nhích nhẹ (+1–2%, trong nhiễu): `DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE=1`, `DS4_METAL_DISABLE_STREAMING_EXPERT_MASKED_ADDR=1`, `DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS=1`, `DS4_METAL_DISABLE_M5_IQ2_PAIR_PACK2=1`, `DS4_METAL_DISABLE_MOE_MM_ID_PAIR_SWIGLU=1`, `DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION=1` (không cộng dồn rõ ràng; dao động ±8% giữa các lần chạy).

**Đã thử và KHÔNG hiệu quả:** lookahead K=2/3/4 (13.7/12.1/11.1 — tệ hơn K=1), tắt readahead (14.5), tắt pread QoS, cache 5200+, `--ssd-streaming-full-layers` (bỏ qua cho V4.1), `-t 10`, ctx 512/1024 (14.7/13.9), nr0 tile IQ2_XXS 2/8 (không đổi).

## 4. Phân tích roofline (tại 16.1 t/s = ~62 ms/token)

Số đo từ timing summary của fork (256 token, cache 4900):
- Hit rate **88.1%**, miss đọc **~289 MB/token**; `pread 12.9 ms + readahead 6.1 ms + bind 12.3 ms + prepare 3.1 ms` — phần lớn **overlap** với GPU (worker thread).
- GPU: **resident weights ~26 ms** (7.45 GiB/token @ ~290 GB/s) + **MoE ~25 ms** (2.04 GiB expert @ ~82 GB/s = 3.5× sàn) + nhỏ khác.
- Tổng: GPU ~51 ms + host không kịp giấu ~11 ms ≈ 62 ms.

**Vì sao 20 t/s khó:** 20 t/s = 50 ms/token; sàn tuyệt đối của fork (`docs/V41_64GB_BUILD.md` §2.1) là 54–66 ms ≈ 15–19 t/s khi cache hit 93–98%. Ta chỉ còn 2 đòn lớn:
1. **MoE kernel**: 25 ms → ~12 ms (đạt ~160 GB/s) ⇒ ~20.5 t/s. Đây là việc Metal kernel thật sự (IQ2_XXS addr-mask pair swiglu / sum6), rủi ro cao, cần một buổi chuyên biệt + oracle byte-identical.
2. **`attn_output_a/b` Q8_0 → Q4_K** (3.2 GiB resident): fork bắt buộc Q8_0 vì kernel "grouped output" chỉ đọc block Q8. Cần viết kernel grouped-output cho Q4_K (validator + kernel), tiết kiệm ~5 ms/token.
3. Hạ `token_embd` F16 (1.23 GiB): fork bắt buộc F16 ở validator; kernel gather-row có thể dùng Q8_0/Q4_K — cần sửa tương tự.
4. Nới wire-limit (`vm.user_wire_limit` 52.48 GiB, không sudo) để cache > 42 GiB — lợi ích nhỏ (hit 88→89%).

## 5. DSpark (bước 7)

- Support GGUF đã build: `~/.local/share/ai-gateway/ds4-models/ds41f-dspark-support.gguf` — **8,633,942,016 B** từ 3 shard MTP native (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, rev `1bae0e10`, 2401 tensor `mtp.*`, block 5, target [37,38,39]) bằng `gguf-tools/deepseek41_dspark.py`.
- **Chạy được** (sau khi phát hiện fork yêu cầu `down_exps = Q2_K` cho batch verification streaming): output đúng, `errors=0`, accept_rate 84.8–87%, avg_accept 1.5–2.6.
- **Nhưng chậm hơn plain**: 9.1 t/s (conf mặc định) / 6.5 t/s (conf 0) vs 16.1 plain — drafter chỉ accept ~45% mỗi token khi propose đủ 5, verify 6-row qua SSD tốn.
- Hiện tại `ffn_down_exps` đã chuyển về **IQ2_XXS** để tối ưu plain decode ⇒ đường batch selected-addr của DSpark không còn bật (đổi lại: plain +~0.3 t/s và cache +1%). Muốn thử DSpark lại: đổi down về Q2_K (script `/tmp/fix_down.py` — cần lưu lại nếu dùng tiếp) và thêm `DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR=1`.

## 6. Việc nên làm tiếp (đề xuất thứ tự)

1. **Viết lại kernel MoE decode** (IQ2_XXS pair-swiglu + sum6, 1 token): mục tiêu 160+ GB/s. Đây là con đường duy nhất tới ~20 t/s trong ràng buộc phần cứng.
2. Nếu (1) xong mà vẫn thiếu: kernel grouped-output Q4_K cho `attn_output_a/b` + gather-row Q8 cho `token_embd`, mở khoá ~1.5 GiB resident nữa (~5 ms).
3. DSpark: cần drafter tốt hơn (checkpoint MTP gốc chỉ ~45–50%/token trên model này) hoặc chấp nhận plain.
4. Đo lại tại ctx 8K/32K khi dùng thật (t/s sẽ giảm về ~13–14 @8K như doc của fork).

## 7. Trạng thái hệ thống & vệ sinh

- Prod 35B: `server-power.sh on` → backend ds4 :18087 + gateway :8090, model `Shiftedx--ornith-1.5-35b-a3b-abliterated-attention8-bf16recurrence-vision-mtplx`, KV-disk `ds4-kv-cache-ornith` nguyên vẹn.
- `ds4.c`, `ds4_metal.m`, `metal/*.metal`: **đã revert về nguyên trạng** (mọi thí nghiệm kernel/nr0/budget đã undo; git tree sạch ngoài file report này).
- Model file: 1010 tensor / 130 KV, mọi dtype hợp lệ với validator V4.1, bảng Engram page-aligned, `resident_end == tbl1`.
- Commit chuỗi tool: `9ba11be6`, `835a7593`, `3a685a04`, `e3e107a2` (+ các commit merge/grow/fix trước đó).
- Dữ liệu phụ: `ds41f-dspark-support.gguf` (8.6 GB), `ds41f-dspark-src/` (3 shard MTP 7.9 GB) — có thể xoá nếu cần chỗ (tải lại từ HF).
## 8. Kernel work (đêm 2026-10-27) — kết quả & kết luận

Đo chính xác hơn bằng `DS4_METAL_GPU_BUSY_PROFILE` (decode-only, `-n 128`):
- Wall ~82 ms/token, **GPU busy chỉ 25.5 ms (31%)** ⇒ decode **bị chặn bởi host/latency**, không phải GPU.
- Host path (timing summary, cache 4900): pread 12.9 ms + readahead 6.1 ms + "selected" bind wrapper 10.4 ms + prepare 3.1 ms ≈ 32 ms/token, phần lớn **serial** với GPU vì mỗi layer phải chờ router→load expert.
- Preads ĐÃ song song (tối đa 9 thread). Lookahead K=1 là tối ưu (K≥2 tệ hơn). Đường **split-deferred** (submit resident stage trước, load missing expert bằng CPU trong lúc GPU chạy) chỉ bật khi `g_batch_cb != nil` và không ở chế độ profile — không kích hoạt trong decode CLI/server single-stream ⇒ đây là chỗ còn ~30-50 ms/token tiềm năng (việc lớn tiếp theo: mở overlap cho decode thường).

Kernel MoE (thử nghiệm):
- Viết `kernel_mul_mv_iq2_xxs_pair_f32_impl_fast` + `kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_fast_f32` (grid load 1× uint64, sign gộp vào thanh ghi y, FMA) — bật bằng `DS4_V41_MOE_FAST=1`.
- Kết quả: **+3% (15.3 → 15.8 t/s)**, output **byte-identical** trên 3 prompt oracle (md5 trùng). Mặc định vẫn dùng kernel gốc.
- Bản fast cho kernel **down** làm lệch bit (thứ tự cộng bị tái kết hợp) → đã bỏ; đây là lý do chỉ giữ fast cho gate/up.
- Lưu ý: fast pair là opt-in; nếu dùng cho production cần chạy bộ quality 112 prompt của fork để xác nhận byte-identical.

Kết luận: 20 t/s cần **pipeline hoá host path** (overlap load expert với GPU giữa các layer), không phải micro-opt kernel. Ước tính: nếu overlap được toàn bộ host (32ms) với GPU (25ms) → ~35-40 ms/token ≈ 25-28 t/s (trần mới), trong khi micro-opt kernel chỉ cho +3%.
