# Review Strata (Niko1221/Strata) — học gì cho ds4-metal + gateway (2026-10-28)

Repo: <https://github.com/Niko1221/Strata> — engine C++/Metal/CUDA cho đúng
Qwen3.8-Flash-Next (và DeepSeek V4.1), OpenAI/Anthropic API localhost, ~34.7k sao,
MIT. Có cả Mac (Metal), CUDA, SYCL.

Mục tiêu: tìm cách cải thiện **prefill** và **khả năng đọc** của stack hiện tại
(ds4-metal trên M5 Pro 20 core, 64 GB).

## 1. So sánh hiệu năng — không có khoảng cách kernel

| | Strata (M5 Max 40 core, 128 GB) | ds4 (M5 Pro **20 core**, 64 GB) |
|---|---|---|
| Prefill 4K / 32K | 1.430 / 1.460 tok/s (Q2_0) | 732 / 787 tok/s (IQ2_XXS, chunk 4096) |
| Decode (MTP) | 119,5 / 108,2 tok/s (`--spec 2`) | 50,1 tok/s (auto depth 2↔3) |
| Decode (không MTP) | 87,4 tok/s | ~? (MTP đang bật) |

Chuẩn hoá theo GPU core: Strata **35,8 tok/s/core**, ds4 **36,6 tok/s/core**.
⇒ Kernel prefill của ds4 **không thua**; port kernel (MoE MM, GDN, QSA) từ Strata
gần như chắc chắn không đem lại gì. Đừng đuổi hướng đó.

Đối chiếu nội bộ cũng khớp: profile ds4 cho MoE 37%, GDN 26%, attention 20%,
hc 16%, PLE 1% — và GDN của Strata cũng là **scan tuần tự theo token**, song song
theo cột state (kernel `gdn_prompt.metal`: 3 pass `gdn_pre`/`gdn_scan`/`gdn_post`,
giống hệt cấu trúc `gdn_front`/`gdn_prep`/`gdn_scan`/`gdn_out` của ds4). Không có
thuật toán mới ở đó.

## 2. Kỹ thuật của Strata — cái nào áp dụng được

| Kỹ thuật (doc) | Strata | Ta | Áp dụng |
|---|---|---|---|
| **KVA** — đọc prompt chính xác các layer đầu, layer cuối chỉ dựng state (GDN recur + KV pages + indexer), bỏ attention-out/out-proj/MoE; đọc chính xác 2048 token cuối (`docs/KVA.md`) | opt-in, +11–18% @14–30K, +28% trên R9700 27K | **chưa có** | **Ứng viên #1** |
| **Prefill head alias** — mượn VRAM của output head cho prompt buffer (`docs/PREFILL_HEAD_ALIAS.md`) | +chunk 2048 trên card 8 GB | chưa | Thấp (unified memory, head nhỏ) |
| **Message-boundary cache** — checkpoint trước tin nhắn cuối khi user sửa (`docs/MESSAGE_BOUNDARY_CACHE.md`) | -33% first-edit | **đã có tương đương** | Không cần |
| **Prompt cache tail** — checkpoint sát cuối prompt (`docs/PROMPT_CACHE_TAIL.md`) | giảm re-read khi branch | **đã có** (cold checkpoint) | Không cần |
| **KV prefetch** — upload KV song song compute (`docs/KV_PREFETCH.md`) | +4–5% prefill khi KV offload RAM | N/A (KV không offload) | N/A |
| **ExpertDec** — kernel decode expert bằng shared-mem (`docs/EXPERTDEC.md`) | CUDA-only, bit-exact | N/A (Metal) | N/A |
| **Batch slots** — nhiều request/lần (`docs/BATCHING.md`) | +31–39% aggregate với 2–4 slot | 1 slot (RAM) | Chỉ Ornith nếu cần đa người dùng |
| **Spec depth + min-p** | `--spec 2/3`, `--spec-min-p 0.5` | auto 2↔3 (đo: depth 3 **−12%**) | Không cần |
| **Chunk Mac** | 4096 (≥96 GB), 1024 (máy nhỏ) | 4096 @64 GB | Phù hợp |
| **Quant quality** | IQ3_S 87,9% top-1, UD-Q4_K_XL 92,8% (vs Q8_0) | IQ2_XXS chưa đo | Không đổi được (RAM) |

### Đã kiểm chứng lại bằng đo trên máy này

- **Sửa tin nhắn cuối**: prompt 20.296 token, đổi 13 token cuối → request 2 chỉ
  4,0 s (vs 31,1 s) và backend báo `common=20283/20296`. Tức cold checkpoint
  (trim 1864) đã phủ đúng case "edit/retry" mà Strata thêm checkpoint riêng.
  Không cần làm gì thêm.
- **Spec depth**: depth 3 = 44,1 tok/s, depth 2 = 50,0, auto = 50,1. Auto thắng.

## 3. Đề xuất (xếp theo kỳ vọng/công sức)

### P1 — KVA cho prefill dài (kỳ vọng +12–18%, rủi ro chất lượng có kiểm soát)

Ý tưởng port vào ds4: trong prefill, với các layer cuối (ví dụ 36–47/48; model ta
có 49 block = 48 trunk + MTP):
- vẫn tính QKV projection + **ghi KV page** và **cập nhật GDN state** (để decode
  chạy đúng),
- **bỏ** attention output, out-proj, router + MoE + shared expert của các layer đó,
- đọc chính xác N token cuối (Strata dùng 2048) để "neo" state.

Vì MoE 37% + attn 20% + hc 16% = 73% thời gian, bỏ 25% số layer ≈ **−15–18%**
prefill, trừ phần state bắt buộc → kỳ vọng thực **+10–15%**.

Gating bắt buộc trước khi bật mặc định:
- needle 52K/128K (harness `/tmp/needle2.py`) phải giữ 3/3,
- một bộ task extraction/agent (Strata dùng 22 task; split 24 làm hỏng, split 36 sạch),
- so decode t/s sau KVA (Strata thấy decode giảm ~11% trên GPU nhanh).

Nếu làm, đặt sau env `DS4_QWEN4_KVA_SPLIT` / `..._TAIL`, mặc định **tắt**.

### P2 — Readahead expert cho ds41f (streaming SSD)

Strata giữ expert "hot" trong arena và **đọc trước expert của layer kế tiếp**
(`docs/MAC.md`: "read-ahead for the next layer's likely experts"). ds41f của ta
đang SSD-stream 377 GiB với cache 2600 slot và prefill chỉ 115–207 t/s — đây là
chỗ kỹ thuật này có thể đem lại nhiều nhất, vì mỗi layer đang chờ SSD.

### P3 — Batch 2 slot cho Ornith (chỉ khi cần nhiều người dùng)

Strata đo +31–39% aggregate với 2–4 slot. Ornith 28 GiB + 2×KV có thể vừa 64 GB,
Qwen 53 GiB thì không. ds4 đã có đường batch (`ds4_qwen4_*_batch_rows`), nhưng
chưa mở cho gateway.

### P4 — Đo chất lượng quant hiện tại

Strata công bố top-1/KL vs Q8_0 cho từng pack. Ta chưa có số cho
`IQ2XXS-Q2KDownPad768-DenseQ4Kselimat`. Nếu sau này có RAM, IQ3_XXS/Q2_0 là bậc
chất lượng kế tiếp (không vừa 64 GB bây giờ).

## 4. Không nên đuổi

- Port kernel MoE/GDN/QSA: hiệu năng/core đã ngang.
- `pcie_frac`, `--vram-reserve-mib`, head alias: thuộc kiến trúc VRAM rời.
- Tăng spec depth: đã đo là chậm hơn.

## 5. Nguồn

`docs/MAC.md`, `docs/KVA.md`, `docs/PREFILL_HEAD_ALIAS.md`,
`docs/MESSAGE_BOUNDARY_CACHE.md`, `docs/PROMPT_CACHE_TAIL.md`,
`docs/KV_PREFETCH.md`, `docs/EXPERTDEC.md`, `docs/BATCHING.md`,
`metal/kernels/gdn_prompt.metal`, `metal/kernels/moe_mm.metal`.
Số của ta: `docs/Task/2026-10-28-qwen38-prefill-reading.md`.