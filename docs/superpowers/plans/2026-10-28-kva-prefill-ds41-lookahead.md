# Plan — KVA prefill cho Qwen3.8 + lookahead cho ds41f (2026-10-28)

Mục tiêu: đưa prefill prompt dài của ds4 **vượt Strata** (hiện ngang nhau/core:
36,6 vs 35,8 tok/s/core) mà không đổi model, không giảm chất lượng đọc; và tận
dụng nốt lookahead prefetch cho ds41f.

Cả hai đều **opt-in, mặc định tắt**, chỉ bật khi qua cổng kiểm định.

## Phase 0 — Source parity (bắt buộc trước mọi thay đổi)

Vấn đề: binary Qwen đang chạy (`~/.local/share/ai-gateway/ds4-metal/ds4-server`,
12/10) chứa các knob **không có trong bất kỳ cây nguồn nào trên máy** hoặc trên
các branch public (`GDN_PREFILL_NSG`, `MOE_MM_NAX`, `ATTN_MERGE_WIDE`). Nếu build
lại từ `ornith-122b` mà mất các tuning đó, ta có thể chậm đi 5–10% trước khi KVA
kịp bù.

Việc:
1. Build `ds4-server` từ các cây ứng viên: `ornith-122b`, `develop`, `foxface`,
   `~/Documents/GitHub/ds4-metal` (mỗi build ~2 phút).
2. Đo mỗi binary: prefill 24k lạnh + decode 160 token, cùng prompt (script
   `scripts/bench-prefill.py`, `bench-decode.py` chuyển từ `/tmp`).
3. Chọn cây nhanh nhất; yêu cầu **≥98%** của 732 t/s / 50,1 t/s.
4. Nếu không cây nào đạt: dừng, truy nguồn binary 12/10 (git reflog các worktree,
   `~/.local/share/.../.phase1-backup-*`) trước khi sửa kernel.

## P1 — KVA: prefill xấp xỉ layer cuối

Tham chiếu: Strata `docs/KVA.md` (split 36/48, tail 2048; +11–18% @14–30K,
+28% trên R9700 27K; split 24 làm rơi fact, split 36 sạch).

### Thiết kế

- Env mới (đọc 1 lần lúc mở graph):
  - `DS4_QWEN4_KVA_SPLIT=N` — 0 = tắt (mặc định); N = layer đầu tiên bị xấp xỉ
    (đề xuất 36 với 48 layer trunk).
  - `DS4_QWEN4_KVA_TAIL=T` — số token cuối đọc **chính xác** (mặc định 2048,
    clamp 512..8192).
- Chính sách chunk: chunk `[pos0, pos0+T)` bị xấp xỉ khi
  `KVA_SPLIT > 0 && pos0 + T <= L - TAIL` (L = độ dài prompt). Các chunk cuối
  phủ tail chạy đầy đủ.
- Trong vòng lặp layer (`ds4.c` ≈63392):
  - `il < split`: chạy như hiện tại.
  - `il >= split` và chunk xấp xỉ:
    - giữ `qwen4_graph_hc_mix` (attn norm) — cần cho projection,
    - **attention layer**: gọi phần `attn_prep` (qkv, rope, ghi KV, indexer keys),
      **bỏ** `qwen4_graph_attention_core` + out-proj + hc_combine,
    - **linear (GDN) layer**: `gdn_front` + `gdn_prep` + `gdn_scan` (cập nhật state),
      **bỏ** `gdn_out` + hc_combine,
    - **bỏ** `hc_ffn_mix` và `qwen4_graph_moe`.
  - R (hidden) giữ nguyên giá trị của layer `split-1`; mỗi layer cuối tính K/V và
    state từ chính R đó — đúng định nghĩa xấp xỉ của KVA.
- Tail chạy đầy đủ nên decode khởi động từ state chính xác ở các vị trí cuối.

### Điểm sửa

| File | Hàm | Thay đổi |
|---|---|---|
| `ds4.c` | `qwen4_graph_attention_tail` (≈62903) | tách nhánh `prep_only` trả về sau `attn_prep` |
| `ds4.c` | `qwen4_graph_linear` (≈62503) | tham số `state_only`: dừng sau `gdn_scan` |
| `ds4.c` | vòng lặp prefill (≈63392) | nhánh `approx`, bỏ combine/MoE |
| `ds4.c` | env parse + log 1 dòng khi bật | `KVA split=36 tail=2048` |
| `ds4_server.c` | — | không cần (env process-wide) |

Không thêm buffer mới ⇒ memory plan không đổi.

### Kỳ vọng

`(48-36)/48 × (MoE 37% + attn-out ~8% + hc ~6%)` ≈ **+12–14%** prefill; tail
2048 tốn ~1 chunk/13 ⇒ net **+10–13%**. Tại 52k: 74 s → ~66 s.

### Cổng kiểm định (phải qua hết mới bật)

1. Needle 52k và 128k: **3/3** (bằng baseline), dùng `scripts/needle.py`.
2. Bộ extraction 10 prompt tài liệu dài (facts, bảng, code) — không giảm.
3. 5 task agent ngắn (prompt ~10k) — không giảm.
4. Decode sau prompt 52k KVA ≥ **95%** baseline (Strata thấy −11% trên GPU nhanh).
5. KVA tắt ⇒ output **byte-identical** với build hiện tại.

### Sweep

`split ∈ {32,36,40,44} × tail ∈ {2048,4096}`; ghi prefill 24k/52k/128k, decode,
và 5 cổng trên. Chọn điểm qua cổng có prefill cao nhất.

## P2 — ds41f lookahead prefetch

Hiện trạng: **đã có** (`ds41_la_post_layer`, `ds4.c` ≈43695; k mặc định 1;
`DS4_METAL_V41_LOOKAHEAD_K`, `DS4_METAL_DISABLE_V41_LOOKAHEAD`), thiết kế cho
decode (11,35 → mục tiêu 12 t/s).

Việc:
1. Xác nhận `ds41_la_post_layer` có chạy trong **prefill chunk** (T>1) hay chỉ
   decode (`T==1`). Nếu chỉ decode → thêm post cho prefill (job dùng FFN input
   của layer L, hợp lệ theo chunk).
2. Đo ds41f với k = 0/1/3/6: prefill 8k/30k, decode 4k/30k; đọc counter
   `posted/dropped/predicted/issued/used`.
3. Nếu `dropped` cao hoặc SSD idle trong lúc GPU chạy: nâng mailbox lên 2 job
   và/hoặc prefetch thêm layer L+2.
4. Chấp nhận nếu prefill hoặc decode tăng ≥5%, output không đổi (lookahead chỉ
   làm ấm page cache → bit-identical).

## Rollout

- Registry Qwen: thêm env KVA **chỉ khi** cổng qua; nếu không, để trống và lưu
  kết quả.
- Registry ds41f: thêm `DS4_METAL_V41_LOOKAHEAD_K=<best>`.
- Docs: cập nhật `docs/Task/2026-10-28-qwen38-prefill-reading.md` và
  `docs/Task/2026-10-28-strata-review.md` bằng số cuối.
- Rollback: bỏ env, restart backend (không đổi code).

## Ước lượng

| Phase | Thời gian |
|---|---|
| 0 — source parity + harness | 0,5 ngày |
| P1 — implement + sweep | 1,5–2 ngày |
| P1 — validation 5 cổng | 1 ngày |
| P2 — audit + tune | 0,5 ngày |
| **Tổng** | **3,5–4 ngày** |

## Rủi ro

1. **Source parity** (lớn nhất): binary 12/10 có tuning từ cây không còn trên đĩa.
   Giảm thiểu: bisect 10 worktree, hoặc lấy delta làm baseline mới và yêu cầu KVA
   thắng >10%.
2. **Chất lượng KVA** trên extraction/agent: split 24 từng làm rơi fact. Giảm
   thiểu: cổng 1–3 + mặc định tắt.
3. **Drift số học**: nhánh state-only phải không được chạm vào đường chính xác
   khi tắt KVA (cổng 5 bắt buộc).