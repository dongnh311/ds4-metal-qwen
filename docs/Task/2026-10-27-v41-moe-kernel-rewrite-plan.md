# Kế hoạch viết lại kernel Metal cho MoE (V4.1 122B) — mục tiêu 20 t/s

Ngày: 2026-10-27. Trạng thái xuất phát: **15.65 t/s** (step 60.5 ms/token).

## 1. Cơ sở số liệu (profile thật, `DS4_V41_DECODE_PROFILE=1`)

```
step_ms=60.5  gpu_busy_ms=45.5  pread_ms=7.8  engram_ms=0.9  host~6
GPU 45.5 = resident ~21.5 (6.2 GiB @ ~295 GB/s = ĐỈNH) + MoE ~24 (2.39 GB @ ~100 GB/s)
```

Suy ra: chỉ còn một nguồn lợi lớn = **MoE kernel**. Nếu MoE đạt 200 GB/s:
pair 36.5 MB/layer → 0.18 ms, down 23.2 MB/layer → 0.116 ms ⇒ 0.30 ms/layer × 40 = **12 ms**
⇒ step ≈ 48.5 ms ⇒ **~20.6 t/s**. Đây là mục tiêu của kế hoạch.

## 2. Kernel hiện tại (đường decode đang chạy)

| Kernel | Vai trò | Cấu hình |
|---|---|---|
| `kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32` | gate+up IQ2_XXS | grid 288×1×6, TG 64 (32×nsg2), `nr0=4` row/simdgroup, mỗi lane giữ 1 dải K 32-block |
| `kernel_mul_mv_addr_q2_K_sum6_masked_f32` | down Q2_K, cộng 6 expert trong kernel | grid 288×1×1, TG 64, `nr0=4`, mỗi lane 5 block |

Điểm yếu đã xác định (đo + đọc code):
1. **Load vô hướng**: mỗi lane đọc `yl[0..31]` bằng 32 lệnh float rời; IQ2_XXS `q[0..3]` 4 lệnh uint16; Q2_K `qs[0..3]` 4 lệnh uint16 + 4 byte `sc` + 2 half `d`.
2. **MLP thấp**: metadata của `nr0` hàng được load *tuần tự trong* vòng row (load → tính → load), không có prefetch; chuỗi phụ thuộc che mất băng thông.
3. **Shared-memory traffic**: IQ2_XXS đọc grid 8 byte/lần bằng 8 lệnh byte (`gridg[j]`), thay vì 1 uint64 + shift.
4. **Chỉ 288 TG cho down** ⇒ ít song song ở cấp threadgroup (6 expert lặp trong cùng TG).

## 3. Nguyên tắc thực thi

- **Cổng bit-exact**: mọi thay đổi phải giữ md5 đầu ra greedy (temp 0, 3 prompt, n=64) trùng baseline. Vector hoá load / prefetch / đọc grid bằng uint64 **không đổi thứ tự cộng** ⇒ phải bit-exact. Nếu một biến thể buộc phải đổi thứ tự (ví dụ tách 6 expert ra 6 TG) thì để **opt-in qua env**, không đổi mặc định.
- **Đo**: mỗi biến thể = 2 run `-n 512` (lấy `generation t/s`) + `gpu_busy_ms`. Chỉ nhận khi ≥ +2%.
- **An toàn**: 1 process/lần (lock `/tmp/ds4.lock`), hạ stack khi benchmark, dựng lại `server-power.sh on` + kiểm tra :8090 cuối mỗi phiên. Build `-j8` (không đụng convert/đa worker).
- **Commit** từng biến thể được nhận, kèm số đo trong message.

## 4. Pha công việc

### Pha B — vector hoá + prefetch (bit-exact) ← làm trước
- B1 `pair`: `yl` bằng 8× `float4`; đọc grid bằng 1 `uint64` + shift thay 8 byte; gom load metadata của 4 hàng ra trước vòng tính (prefetch array `aux32g/aux32u/dg/du`).
- B2 `down sum6`: `yl` 8× `float4`; `qs[0..3]` = 1 load 8 B; `d,dmin` = 1 load half2.
- Kỳ vọng: 120–160 GB/s ⇒ 17–18.5 t/s. Cổng: md5 trùng.

### Pha C — tăng MLP / song song TG (bit-exact)
- C1 `pair`: double-buffer metadata block kế tiếp (prefetch `ib32+32`) trong lúc tính block hiện tại.
- C2 `pair`: thử `nsg=4` (16 hàng/TG, 4 warp/TG) — mỗi hàng vẫn 1 simdgroup, thứ tự cộng không đổi.
- C3 `down`: xử lý 2 block/iteration để tăng load in-flight.
- Kỳ vọng: +10–25% trên nền B.

### Pha D — nếu C chưa đủ (được phép lệch bit, opt-in)
- D1 `down`: tách 6 expert ra 6 TG (z=6) + pass cộng 6 phần tử (thêm 1 kernel nhỏ) ⇒ 1728 TG thay 288; đổi thứ tự cộng ⇒ env `DS4_V41_MOE_DOWN_SPLIT6=1`.
- D2 `pair`: gộp 2 expert/TG để chia sẻ `yl` (giảm L1/L2 traffic).
- Kỳ vọng: 18.5 → 20+.

### Pha E — top-up host (nếu thiếu ≤1 t/s)
- E1 Bỏ ghi lại bảng địa chỉ expert khi tập expert của layer không đổi so với token trước (fast path trong `bind`, tiết kiệm một phần 10.4 ms).
- E2 Giảm miss: loại `token_embd` khỏi resident (1.3 GB) để tăng slot cache (17.9 miss/token ⇒ pread 7.8 ms).

## 5. Bảng mục tiêu

| Mốc | step ms | t/s |
|---|---|---|
| hiện tại | 60.5 | 15.65 |
| sau B | ~54 | 18.5 |
| sau C | ~51 | 19.6 |
| sau D/E | ≤50 | **20+** |

## 6. Rủi ro

- Metal không cho vector load misaligned ⇒ kiểm tra alignment từng trường (đã tính: `y4` 128 B, `qs` 8 B, `d` 4 B đều aligned; `sc` chỉ 1 B nên giữ load byte).
- Kernel sai ⇒ lỗi im lặng: luôn chạy oracle md5 + 3 prompt chất lượng trước khi nhận.
- Nhiệt/tiến trình: không chạy song song nhiều `ds4`; nghỉ giữa các run dài.
## 7. Kết quả đo sau Pha B (2026-10-27) — giả thuyết bị bác bỏ

### 7.1 Trần phần cứng (bench Metal độc lập, `/tmp/bwtest2`)

| Nguồn đọc | Băng thông |
|---|---|
| Buffer private 256 MiB, ≥576 TG × 64 thr | **240–261 GB/s** |
| **mmap file model (shared buffer)** | **258–262 GB/s** |

⇒ Đường bộ nhớ (kể cả mmap expert cache) **không phải** nút thắt.

### 7.2 Các biến thể đã thử (đo bằng `generation t/s`, -n 512, ctx 2048, cache 4900)

| # | Biến thể | Bit-exact | Kết quả | Kết luận |
|---|---|---|---|---|
| 1 | pair: `yl` 8×float4 + grid 1×uint64 + prefetch metadata 4 hàng | ✅ | 15.52–15.56 | trung tính/âm nhẹ |
| 2 | pair: tăng `nr0` 4→8 (IQ2_XXS) | ✅ | 15.52 | trung tính |
| 3 | down: `nr0` 4→8 (Q2_K) | ✅ | 15.60–15.65, down vẫn 0.28 ms | trung tính |
| 4 | down: stage `y` 20 KB vào threadgroup memory | ✅ | **14.43** | −8% do mất occupancy (1 TG/core) |
| 5 | pair: bảng grid float trong threadgroup | ✅ | 15.59–15.62 | trung tính (ALU→LSU shift) |
| 6 | pair: **half2 packed** (2 weight/lệnh), opt-in `DS4_V41_MOE_H2` | ❌ | **15.79–15.81**, gate_up 0.344→0.317 | +0.8% |

### 7.3 Chẩn đoán đúng

Traffic thật của MoE (không chỉ weight):
- pair: mỗi TG đọc **21 KB weight + 40 KB lại phần activation `y`** (mỗi simdgroup đọc trọn hàng y).
- down: **80 KB weight + 240 KB `y`** (2 simdgroup × 6 expert × 20 KB).
- ⇒ mỗi layer ~105 MB / 320 MB ⇒ ở 0.32/0.27 ms = **330 GB/s tổng tải** (trên trần DRAM nhờ `y` hit L2).

Vì vậy: giảm số lệnh, mở rộng load, bảng float/half, tăng nr0 **đều không đổi** — nút thắt là **tổng byte phải tải, trong đó activation bị đọc lặp 2–3× so với weight**.

### 7.4 Thiết kế đúng cho lần tới (Pha B2)

Mục tiêu: `y` chỉ đọc **một lần cho 32 hàng** thay vì một lần cho 4 hàng.

- TG = **128 thread (4 simdgroup) × nr0 = 8** ⇒ 32 hàng/TG; `y` stage **half (10 KB)** trong threadgroup memory (occupancy ~3 TG/core = 384 thread/core, chấp nhận được).
- Grid: pair = 2304/32 × 6 expert = **432 TG** (đủ); down: tách 6 expert sang trục z (432 TG) + 1 kernel cộng 6 partial (đổi thứ tự cộng ⇒ opt-in).
- Traffic/hàng: 3.2 KB thay vì 7.6 KB ⇒ **~2.3×** ít hơn ⇒ MoE 25 ms → ~12 ms ⇒ step ~48 ms ⇒ **~20.5 t/s**.
- Điều kiện: sai số chỉ ở tích half (tổng vẫn fp32) — cần oracle md5 + 3 prompt chất lượng; giữ đường fp32 làm mặc định.

### 7.5 Qwen3.8-flash-next (prod 35B)

Model prod `Ornith-1.5-35B-A3B-…gguf` (41 layer, hidden 2048) có expert routed **Q5_K (45 tensor) + Q4_K (78)**, shared expert Q8_0, không có IQ2_XXS ⇒ **không đi qua kernel vừa sửa**. Muốn "cả hai cùng hưởng lợi" thì thiết kế 7.4 phải áp cho họ kernel `kernel_mul_mv_*_q4_K/q5_K_sum6` và `_pair_swiglu` của nó, kèm **xác nhận md5 bit-exact trên đúng lệnh prod** (KV cache đĩa `ds4-kv-cache-ornith` không được đổi).

## 8. Pha B2 (32/64 hàng mỗi TG + stage `y`) — kết quả

Đã dựng 2 cấu hình (opt-in `DS4_V41_MOE_B2`), cả hai **không thắng**:

| Cấu hình | `gate_up`/layer | generation | Ghi chú |
|---|---|---|---|
| B2-a: 4 simdgroup × nr0=8 = 32 hàng/TG, stage `y` **float** 20 KB | — | 13.95 | 1 TG/core, mất occupancy |
| B2-b: 8 simdgroup × nr0=8 = 64 hàng/TG, stage `y` **half** 10 KB + half2 | 0.344 ms (không đổi) | 15.21–15.32 | traffic/2.5× nhưng thời gian y nguyên |
| Mặc định | 0.344 ms | 15.65 | |

Chất lượng B2-b: output mạch lạc (half chỉ ở tích, tổng vẫn fp32) ⇒ về mặt số học dùng được, nhưng **không có lợi tốc độ** nên đã revert.

**Kết luận quan trọng:** giảm 2.5× traffic activation vẫn **không đổi thời gian** ⇒ nút thắt không phải tổng byte, mà là **hiệu suất đọc weight theo pattern 8 B × stride 66 B** (mỗi warp đọc 4 khối 64 B nằm trong 3 dòng 128 B ⇒ ~30% lãng phí sector + latency L2). Đây là giới hạn của **layout tensor**, không phải của vòng lặp.

## 9. Hướng còn lại (cần một phiên riêng): repack tensor expert sang dạng phẳng

Ý tưởng: tách mỗi expert tensor thành 2 mặt phẳng (SoA) — mảng `d`/scale và mảng **các chunk 8 B liền mạch** — sao cho 32 lane đọc đúng **256 B liên tục** (100% sector) thay vì 8 B × stride 66 B.
- Kích thước tensor **không đổi** (chỉ hoán vị byte trong từng tensor) ⇒ có thể ghi in-place theo từng expert, không cần thêm đĩa.
- Áp dụng cho: IQ2_XXS gate/up (V4.1) và Q4_K/Q5_K gate/up + Q2_K down (35B prod) ⇒ cả hai model cùng hưởng.
- Kỳ vọng: sector utilization ~70% → ~100%, tức 1.4×; cộng giảm latency ⇒ MoE 25 ms → 15–17 ms ⇒ step 51–53 ms ⇒ **19–19.6 t/s**, và mức 20 cần thêm ~1 ms từ host (Pha E).

**Trước khi làm repack phải đo bằng GPU counter** (`MTLCounterSampleBuffer`, L2 hit + sector utilization) để xác nhận giả thuyết lãng phí sector, tránh lặp lại vòng "đoán rồi thử" vừa qua.

## 10. Probe layout (2026-10-27) — bác bỏ giả thuyết lãng phí sector

Nguồn: `docs/Task/2026-10-27-moe-layout-probe.m` (đọc thẳng file model, mmap + shared buffer).
So sánh 2 pattern với **cùng số byte hữu ích** (20×64 B mỗi hàng):

| Pattern | TG | GB/s hữu ích |
|---|---|---|
| **blocked66** (layout hiện tại: 8 B/lane, block stride 66 B) | 1728 | **258.3** |
| blocked66 | 3456 | 268.1 |
| planar128 (1 lane = 8 B, warp = 256 B liền mạch) | 1728 | 267.3 |
| planar128 | 288 | 272.3 |

⇒ Layout hiện tại **đạt 258 GB/s**, gần bằng planar. **Không cần repack 150 GB tensor** (tiết kiệm rủi ro lớn), và nút thắt không nằm ở pattern đọc.

## 11. Tổng kết: 8 giả thuyết đã kiểm, tất cả bị bác bỏ

| # | Giả thuyết | Kiểm bằng | Kết quả |
|---|---|---|---|
| 1 | Thiếu băng thông DRAM | bench streaming | trần 240–262 GB/s, không phải nút thắt |
| 2 | Đường mmap/expert cache chậm | bench mmap file | 258–262 GB/s |
| 3 | Quá nhiều lệnh ALU | vector load + grid + prefetch | trung tính |
| 4 | ALU thừa (byte extract) | bảng grid float | trung tính |
| 5 | ALU fp32 chậm | half2 packed | +0.8% |
| 6 | `y` bị đọc lặp | nr0 8; stage smem 20 KB; 64 hàng/TG + half 10 KB | trung tính / chậm hơn |
| 7 | Layout block 66 B lãng phí sector | layout probe | 258 GB/s ⇒ bác bỏ |
| 8 | Ít threadgroup | 288/1728/3456 TG | đã đủ ở 1728 |

**Kết luận:** kernel MoE không còn dư địa qua các đòn hiện có. Chênh lệch giữa "258 GB/s khả dụng" và "~107 GB/s weights thực tế" nằm ở **cấu trúc engine**: mỗi layer phải chờ host đọc router id → cập nhật bảng expert → pread expert thiếu → encode, và GPU bị ngắt bởi các điểm đồng bộ này (step 60.5 ms trong khi GPU busy 45.5 ms; ~15 ms host + các khoảng trống giữa layer).

**Đường còn lại để tới 20 t/s = tái cấu trúc pipeline decode của engine** (không phải kernel):
1. Bỏ đồng bộ router per-layer: đọc selected id của layer L **trong lúc** GPU đang chạy FFN của layer L-1 (double-buffer router buffer).
2. Prefetch expert của layer L+1 ngay khi có router id (đã có lookahead K=1; mở rộng K=2 với hàng đợi pread riêng).
3. Gom command buffer: 1 CB cho nhiều layer thay vì commit/wait từng layer.
Mục tiêu: step 60.5 → ~50 ms (host 15 → 5, GPU vẫn 45.5).

## 12. Pha E — đo tầng engine (2026-10-27): 15 ms "host" thực chất là gì

### 12.1 Readback per-layer: KHÔNG phải nút thắt
`DS4_METAL_SELECTED_PROFILE=1` (mode `stream-addr-mask`, `ids=override`):
```
read=0.000 ms   bind=0.002 ms (hit) … bind=0.3–0.7 ms (có miss)
```
⇒ Router readback đã được thay bằng lookahead override, **không có đồng bộ đọc id** ⇒ việc "double-buffer router" trong kế hoạch **không cần thiết**.

### 12.2 `bind` = copy expert vào cache
- Trung bình **0.31–0.39 ms/layer** ⇒ **12.5–16 ms/token** (đúng phần chênh `step 60.5 − GPU 45.5`).
- ~1 miss/layer (5/6 expert hit), mỗi miss copy **6.6 MB** (gate+up 2.9 + down 3.7) ở ~17–21 GB/s.
- `DS4_METAL_STREAMING_EXPERT_PREAD_THREADS` 9 → 18: **không đổi** (15.68 vs 15.64).
- Chính sách thay slot: `EVICT_LRU=1`, `HOTNESS_DECAY=4/64` → **misses y nguyên 5567**, bind 2136–2315 ms ⇒ cache đã chạm trần, policy không giúp.

### 12.3 Zero-copy từ mmap: KHÔNG dùng được
`DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE=1`: không hoàn thành nổi 4 token trong 240 s (và timeout 30 phút với n=512) ⇒ page cache của OS không giữ nổi working set ⇒ **cache mlock là bắt buộc**.

### 12.4 Kết luận cuối cùng
| Thành phần | ms/token | Bản chất |
|---|---|---|
| GPU busy | 45.5 | resident 21.5 (đỉnh băng thông) + MoE 24 (kernel đã hết dư địa) |
| **Expert refill** | **12.5–16** | copy ~1 expert/layer vào cache (264 MB/token @ ~17–21 GB/s) |
| pread/engram/host khác | ~4 | |

Working set thật: 40 layer × 384 expert × 6.6 MB ≈ **100 GB**; cache khả dụng 41.6 GB (trần mlock 52.48 GiB) ⇒ **không thể khử refill** trên máy 64 GB này.

**Trần thực tế: ~15.6–15.7 t/s.** Để 20 t/s cần ~100 GB cache (máy 128 GB) hoặc giảm byte/expert:
- down Q2_K → IQ2_XXS: expert 6.6 → 5.7 MB (−14%) ⇒ refill ~−2 ms + MoE down ~−2.5 ms ⇒ ước **16.5–17 t/s**, nhưng mất đường mask/split (+1.5%) và cần requant 59 GB (nhiều giờ).
- Không có tổ hợp nào đạt 20 t/s trên cấu hình hiện tại.
