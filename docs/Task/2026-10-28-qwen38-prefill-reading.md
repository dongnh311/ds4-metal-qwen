# Qwen3.8 prefill + long-context reading — đo và các đòn bẩy (2026-10-28)

Máy: M5 Pro 64 GB. Backend `:18086`, gateway `:8090`. Model
`ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2` + `--ple` sidecar, `-c 262144`,
`--prefill-chunk 4096`, `--mtp`, KV 4-bit in-kernel.

## 1. Prefill tiêu vào đâu (profile `DS4_QWEN4_TIMING=2`)

msec mỗi chunk 4096 token, tại pos 8192–12288:

| Stage | ms/chunk | % |
|---|---:|---:|
| MoE | 1999.4 | 37% |
| GDN (linear attn) | 1379.7 | 26% |
| attention | 1059.9 | 20% |
| hc_attn + hc_ffn | 840.7 | 16% |
| PLE | 26.1 | 1% |

⇒ PLE sidecar **không** phải nút thắt (giả thuyết ban đầu sai). Prefill là
compute-bound ở MoE + GDN + attention.

## 2. Quét knob tăng tốc (cùng prompt 24k lạnh, backend t/s)

| Config | t/s | Δ |
|---|---:|---:|
| baseline | 744.0 | — |
| PREFILL_MODE=safe / max | 745.9 / 745.7 | +0.3% |
| GDN_PREFILL_NSG=1 / 8 | 745.7 / 745.2 | +0.2% |
| MOE_MM_NAX=1 | 745.6 | +0.2% |
| DENSE_NAX=1 | 741.7 | −0.3% |
| MOE_MM_SPECIALIZE=1 | 739.2 | −0.6% |
| MOE_MV_NSG=8 | 735.5 | −1.1% |
| HC_MIX_PREFETCH=1 | 735.4 | −1.2% |
| ATTN_MERGE_WIDE=1 | 735.0 | −1.2% |
| Q8_PREFILL_UNPACK=1 | 734.3 | −1.3% |

Tất cả nằm trong nhiễu ±1.3% ⇒ **không có cửa miễn phí** ở tầng kernel hiện tại.

## 3. Trade-off chunk (cùng prompt 24k lạnh)

| chunk | t/s | memory plan |
|---:|---:|---|
| 4096 | 732.5 | **53.19 GiB** |
| 6144 | 764.8 | 55.93 GiB |
| 8192 | 787.5 | 58.67 GiB |

Chọn 4096 vì 64 GB: 8192 để lại ~0 GiB free. 6144 (+4.4%) cần thêm 2.7 GiB,
đẩy máy vào swap.

## 4. Decode / MTP depth

| Config | decode t/s |
|---|---:|
| auto (2↔3 thích ứng) | **50.1** |
| ép depth=2 | 50.0 |
| ép depth=3 | 44.1 (−12%) |

Policy auto đã tối ưu. "`--spec 4`" của stack GPU không map thành lợi ích ở đây
(engine cap depth 3; depth 3 chỉ thắng khi văn bản gần như tất định).

## 5. Khả năng đọc dài (needle 3 mã ở 15/50/85%, haystack đa dạng)

| Prompt | KV | Kết quả | Thời gian |
|---|---|---|---|
| 5.560 tok (sanity) | 4-bit | 3/3 | 14 s |
| 52.293 tok | 4-bit | **3/3** | 74 s |
| 52.293 tok | Q4 tắt | **3/3** | 78 s |
| 121.943 tok | 4-bit | **3/3** | 181 s (ctx 131072, plan 48.83 GiB) |

- KV 4-bit **không** làm mất recall ở 52k.
- 128k đọc chính xác; 181 s gồm ~150 s prefill + decode.
- Lần đo đầu 0/3 là **lỗi test**: mốc chèn needle tính theo token nhưng vòng
  lặp đếm dòng ⇒ không có needle nào trong tài liệu. Đã sửa và xác nhận 3/3.

## 6. Kết luận & lựa chọn

1. Giữ **chunk 4096 @ 262144** (hiện tại): 53.19 GiB, 732 t/s — an toàn.
2. Nếu use case thực tế **≤128k token**: `-c 131072 --prefill-chunk 8192` cho
   plan ~54.3 GiB (48.83 + 5.5) và **~787 t/s (+7.5%)** — cùng mức RAM, nhanh
   hơn. Đây là đòn bẩy lớn nhất còn lại mà không đổi chất lượng.
3. Muốn nhanh hơn nữa phải đổi **dạng lượng tử expert** (Q4_K thay IQ2_XXS)
   hoặc chấp nhận prefill compute-bound của M5 Pro; không còn knob nào.
4. Phía agent: đọc theo `grep/glob → read offset+limit`, lọc shell bằng
   `tail/head/grep`, giữ context < ~32k — cắt thời gian nạp ngữ cảnh, không đổi
   t/s của model.

## 7. Tái lập

```sh
DS4_QWEN4_TIMING=2 ds4-server ... -c 262144 --prefill-chunk 4096   # stage ms
python3 /tmp/needle2.py q4          # 52k, KV 4-bit
python3 /tmp/needle2.py q4off       # 52k, KV_Q4=0
```