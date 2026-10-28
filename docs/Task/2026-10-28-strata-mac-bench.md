# Strata trên máy này — đo thực tế (2026-10-28)

## Kết luận ngắn

Strata **chưa từng chạy trên macOS ở máy này**. Bản đang dùng là trên PC Windows
(`pc-dik-win`, RTX 3060 12 GB, qua Tailscale). Hôm nay đã build engine Metal từ
source và chạy thử: **24k nhanh hơn ds4, nhưng 52k thì engine chết (Metal OOM)**.

## Môi trường đo

- Mac: Apple M5 Pro, 64 GB, macOS 26.6.2.
- Strata: clone `Niko1221/Strata` (main, 2026-10-27), `./setup.sh --model Q2_0`
  → build Metal engine từ source, tải Q2_0 37,62 GB + bảng PLE 28,80 GB
  (tổng 66,4 GB), draft MTP 0,89 GB. Chạy `serve/server.py` + engine Metal,
  `--spec 2`, `--max-context` 32768/65536/262144.
- ds4: Qwen3.8 IQ2_XXS + KVA split 36/tail 2048, chunk 4096, ctx 262144.

## Số đo

| Phép đo | Strata (Mac, Q2_0) | ds4 (Mac, IQ2_XXS + KVA) |
|---|---|---|
| prefill 24k (≈31k token) | **30,3 s = 1.024 t/s** (lần đầu lạnh 785) | 871 t/s @25k |
| prefill 52k | **chết: Metal OOM ở ~33k token** (exit 5), lặp lại 2 lần ở ctx 65.536 và 262.144 | **897 t/s, 58,3 s** |
| prefill 128k | không chạy được | 863 t/s, 150,9 s |
| prefill 200k | không chạy được | 804 t/s, 253 s, 12/12 fact |
| decode | không đo được (engine không restart sau crash) | 50,1 t/s |
| ctx tối đa chạy được | 262k khai báo, nhưng OOM khi prompt ~50k | 262k, đã đọc 203k token |

Strata trên PC Windows (RTX 3060 12 GB, qua Tailscale) — chỉ để tham chiếu, khác
máy: Qwen q2_0 prefill 16,7k token / 35,2 s = **474 t/s**, decode ngắn 16,8 t/s;
DeepSeek V4.1 prefill 16,7k / 144,4 s ≈ **116 t/s**.

## Thử lần 2 — IQ2_XS (cùng hạng quant với ds4)

Tải `IQ2_XS` (39,23 GB + PLE 28,80 GB = 68 GB), patch `mac_fit.py` để thêm pack
này (setup Mac không liệt kê dù bảng model có). Kết quả: engine **từ chối nạp**:

```
ref: unknown type 29
```

Metal engine không có kernel cho tensor type của IQ2_XS ⇒ không chạy được trên
Mac. Đã xoá 68 GB sau khi thử.

## Vì sao

- Strata tải cả 37,6 GB model "wrap" vào Metal + PLE 28,8 GB; prompt buffer cộng
  KV làm vượt hạn mức Metal trên 64 GB khi prompt dài (52k). Ở 24k vừa đủ.
- Trên máy 64 GB, Strata để lại ~5,3 GiB VRAM trống sau khi load; ds4 dùng
  53,19 GiB tổng và vẫn đọc được 203k token.
- DeepSeek V4.1-Flash: tài liệu Strata ghi rõ **"does not run on a Mac yet"**
  (chỉ NVIDIA/AMD). Vì vậy không thể so DS41F-1017 trên máy này; muốn so phải
  dùng PC Windows (RTX 3060) hoặc máy có GPU rời.

## Ghi chú vận hành

- Đã xoá 66,4 GB file Q2_0 + `Strata-data` sau khi đo; giữ repo
  `~/Documents/GitHub/Strata` (494 MB, engine đã build) để chạy lại nếu cần.
- Đã xoá `ds4-kv-cache` (33 GB, cache tái tạo) và cache pip/CocoaPods/ShipIt/
  orca-updater/com.apple.python/swiftpm/Homebrew + cache Chrome để đủ chỗ.
  **Không đụng** `ds4-kv-cache-ornith` (43 GB, quy định cấm).
- Trong lúc đo phải dừng Qwen (ds4 chiếm 53 GiB); sau đó đã khôi phục backend,
  gateway và watchdog.