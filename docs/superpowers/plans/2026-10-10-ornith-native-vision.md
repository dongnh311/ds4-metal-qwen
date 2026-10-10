# Ornith Native Vision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ornith on ds4 reads images itself (Qwen3-VL encoder, M-RoPE in the qwen35 graph), and the gateway sends Ornith's images to it instead of the 4B describe sidecar.

**Architecture:**
- **ds4.** ds4 already encodes Qwen3-VL images for Qwen3.8 (`DS4_VISION_QWEN4`). Ornith's mmproj is the same encoder with a 2048-wide projection. The change is in the Ornith graph:
  - image embedding rows instead of placeholder-token rows;
  - `(t, h, w)` rope positions with a running offset after each image;
  - the same for the MTP block;
  - the offset saved and restored with the rewind point.
- **Server.** The rewind point is extended to image conversations, so native vision does not take away the re-send/compaction protection shipped on 2026-10-09.
- **Gateway.** The gateway flips the two Ornith routes to `vision: true`. It keeps the sidecar for images ds4 cannot take: more than 16, or not PNG/JPEG.

**Tech Stack:** C/Metal (ds4-metal), Python 3.12 (AI-Gateway-MLX), llama.cpp `b1-311d421` as a reference.

**Spec:** `docs/superpowers/specs/2026-10-10-ornith-native-vision-design.md` (ds4-metal, commits `83a7c6b0`, `2809c702`).

## Global Constraints

- Code, commits and repo docs are in English. Reports to the user are in Vietnamese.
- mmproj: `mmproj-Ornith-1.5-35B-A3B-f16.gguf`, 899,283,296 bytes, sha256 `815c9a671991aceeef46ca1aa103b64499db8cfadf21c83005a20ed839d53c7c`. Spike copy: `~/orca/workspaces/ds4-metal-data/gguf/ornith-mmproj/`.
- Model under test: `~/.local/share/ai-gateway/ds4-models/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`.
- One model process at a time on this 64 GB box. Every model-backed run needs the gateway's ds4 slot held down:
  1. run `python3 ~/Documents/GitHub/AI-Gateway-MLX/scripts/quiet-window.py` (exit 0);
  2. start a flock holder on `/tmp/ds4.lock`;
  3. SIGTERM the slot (never SIGKILL a Metal process);
  4. run with `DS4_LOCK_FILE=<scratchpad>/ds4-test.lock`;
  5. release the holder.

  Template: `scratchpad/vision-spike2.sh` and `scratchpad/hold_ds4_lock.py` from 2026-10-10.
- AI-Gateway-MLX rule #11: no test touches the live stack. Only Task 10 deploys. Re-stage only with `scripts/restage.sh`, then run Tier L.
- AI-Gateway-MLX rule #12: commit only your own paths (`git add <paths>`, never `-A`). Never delete `.git/index.lock`.
- ds4-metal: push only to `origin` (dongnh311/ds4-metal-qwen), never to `antirez` or `upstream`.
- KV cache files and server traces hold prompt text. Never print their content.
- Text-only Ornith behaviour must not change. That means bit-identical logits with no image spans.

## Review Focus

1. **An image inside a tool result** (Claude Code `Read` of a PNG through the Anthropic door) must reach ds4 as an image inside `<tool_response>`, in order. Task 4 pins this.
2. **A seventeenth image, or a GIF/WebP**, must never reach ds4. Those parts go to the sidecar by arrival order, so the prompt prefix stays stable. Task 7 pins this.
3. **An image early in a long session, then a re-send or a Claude Code compaction** must still hit the rewind point under fresh image markers, with correct positions after it. Tasks 3, 5 and 6 pin this.
4. **A live continuation that appends a new image after text turns** must continue the rope offset from the history, not from zero. Task 2 pins this.
5. **A slot that held image state and then loads a text-only disk checkpoint** must reset the offset to zero. Task 3 pins this.

---

## File Structure

ds4-metal (worktree `~/orca/workspaces/ds4-metal/ornith-vision`, branch `feature/ornith-vision`):

| File | Change |
|---|---|
| `ds4.c` | Engine admits `--vision` for qwen35moe. Rewind point saves/restores `mrope_delta` and trims image identities. Test helper `ds4_session_test_read_pos3` |
| `ds4.h` | Declares `ds4_session_test_read_pos3` |
| `ds4_qwen35moe.inc` | Trunk and MTP staging read image rows and M-RoPE positions. `qwen35_mrope_delta_at`. Reset and payload load zero the offset |
| `ds4_server.c` | Rewind point keyed by the marker-normalized text plus image offsets. Mark and hit for image requests. Renderer pin tests |
| `tests/test_qwen35_vision.c` (new) | Model-backed: encode, positions, continuation, decode, MTP, rewind, payload reset, text-only checksum |
| `Makefile` | `test-qwen35-vision` |
| `ds4_help.c`, `docs/SERVER.md` | `--vision` now lists Ornith |
| `speed-bench/ornith-vision/` (new) | Fixtures, E2E script, quality pre-registration, reports, deploy receipt |

AI-Gateway-MLX (worktree `.claude/worktrees/ornith-vision`, branch `dongnh311/ornith-vision`):

| File | Change |
|---|---|
| `gateway/vision.py` | `native_split(obj, limit, types)` |
| `gateway/providers/ds4.py` | `native_image_limit`, `native_image_types` |
| `gateway/server.py` | Phase-1 image selection uses `native_split` for native ds4 routes. The comment at :1771 is fixed |
| `tests/test-suite-vision-native.py` (new) + `tests/run-success-gate.py` | `H.vision_native` |
| `tests/test-suite-vision-capability.py` | ds4 argv oracle and controls |
| `config/runtime-registry.ornith-1.5.json` | `--vision` and `vision: true` on both rows |
| `reports/ornith-vision-2026-10-10/` (new) | Deploy receipt |

---

### Task 1: The engine admits `--vision` for Ornith

**Files:**
- Modify: `ds4.c:74154-74175` (qwen35moe refusal list), `ds4.c:74225-74260` (vision load)
- Create: `tests/test_qwen35_vision.c`
- Modify: `Makefile` (next to `test-qwen35-rewind-point`, lines 133-141)
- Modify: `ds4_help.c`, `docs/SERVER.md` (the `--vision` text)

**Interfaces:**
- Produces: `ds4_engine_open` with `.vision_path` set succeeds for Ornith, and `ds4_engine_has_vision(e)` is true.
- Produces: the test harness skeleton `tests/test_qwen35_vision.c` with `fail()`, `open_engine(bool mtp)`, `fake_image()` and `build_prompt()`. Later tasks add cases to it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_qwen35_vision.c`:

```c
/* Model-backed checks for Ornith native vision.
 *
 * Run with the production ds4 slot held down (one model process on 64 GB):
 *   DS4_TEST_MODEL=<Ornith.gguf> DS4_TEST_MMPROJ=<mmproj> DS4_TEST_IMAGE=<jpeg> \
 *     make test-qwen35-vision
 * DS4_TEST_MTP=1 also opens the embedded MTP block.  Without DS4_TEST_MMPROJ
 * only the text-only checksum case runs (Task 2 compares the two). */
#include "ds4.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CTX 8192

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static ds4_engine *open_engine(bool with_vision, bool mtp) {
    const char *model = getenv("DS4_TEST_MODEL");
    const char *mmproj = getenv("DS4_TEST_MMPROJ");
    if (!model || !model[0]) fail("set DS4_TEST_MODEL to the Ornith GGUF");
    ds4_engine_options opt = {
        .model_path = model,
        .vision_path = with_vision ? mmproj : NULL,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = CTX,
        .prefill_chunk = 64,          /* chunk boundaries fall inside images */
        .glm_mtp = mtp,
        .mtp_draft_tokens = 1,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    return e;
}

static void case_encode(ds4_engine *e) {
    const char *image = getenv("DS4_TEST_IMAGE");
    if (!image || !image[0]) fail("set DS4_TEST_IMAGE to a JPEG or PNG");
    if (!ds4_engine_has_vision(e)) fail("engine has no vision after --vision");
    ds4_vision_embedding emb = {0};
    char err[256] = {0};
    if (!ds4_engine_vision_encode_file(e, image, &emb, err, sizeof(err))) fail(err);
    if (emb.token_count == 0 || emb.grid_width * emb.grid_height != emb.token_count)
        fail("image grid does not cover its rows");
    printf("encode: %u rows (%u x %u)\n", emb.token_count, emb.grid_height, emb.grid_width);
    ds4_vision_embedding_free(&emb);
}

int main(void) {
    const char *mmproj = getenv("DS4_TEST_MMPROJ");
    const char *mtp = getenv("DS4_TEST_MTP");
    const bool vision = mmproj && mmproj[0];
    ds4_engine *e = open_engine(vision, mtp && mtp[0] == '1');
    if (vision) case_encode(e);
    printf("PASS: ornith vision (vision=%d mtp=%d)\n", vision ? 1 : 0, mtp && mtp[0] == '1');
    ds4_engine_close(e);
    return 0;
}
```

Add the following to `Makefile` after the `test-qwen35-rewind-point` rule. Also add `test-qwen35-vision` to the `.PHONY` list on line 72, and `tests/test_qwen35_vision` to the `clean` list on line 1278.

```make
tests/test_qwen35_vision.o: tests/test_qwen35_vision.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen35_vision.c

tests/test_qwen35_vision: tests/test_qwen35_vision.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-qwen35-vision: tests/test_qwen35_vision
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_qwen35_vision > tests/.qwen35-vision-text.out
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MMPROJ="$(DS4_TEST_MMPROJ)" DS4_TEST_IMAGE="$(DS4_TEST_IMAGE)" \
	  ./tests/test_qwen35_vision > tests/.qwen35-vision.out
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MMPROJ="$(DS4_TEST_MMPROJ)" DS4_TEST_IMAGE="$(DS4_TEST_IMAGE)" \
	  DS4_TEST_MTP=1 ./tests/test_qwen35_vision > tests/.qwen35-vision-mtp.out
	grep '^TEXT-CHECKSUM' tests/.qwen35-vision-text.out > tests/.qwen35-vision-a
	grep '^TEXT-CHECKSUM' tests/.qwen35-vision.out > tests/.qwen35-vision-b
	cmp tests/.qwen35-vision-a tests/.qwen35-vision-b
	cat tests/.qwen35-vision.out tests/.qwen35-vision-mtp.out | grep -E '^(PASS|encode|positions|continuation|decode|mtp|rewind|payload)'
```

`TEXT-CHECKSUM` lines appear in Task 2. Until then the two `grep` lines find nothing and `cmp` compares two empty files.

- [ ] **Step 2: Run it to verify it fails**

Hold the slot down per Global Constraints, then run:

```bash
cd ~/orca/workspaces/ds4-metal/ornith-vision && make -j8 tests/test_qwen35_vision > <scratch>/v1-build.log 2>&1 && \
DS4_LOCK_FILE=<scratch>/ds4-test.lock DS4_TEST_MODEL=$HOME/.local/share/ai-gateway/ds4-models/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf \
DS4_TEST_MMPROJ=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith-mmproj/mmproj-Ornith-1.5-35B-A3B-f16.gguf \
DS4_TEST_IMAGE=$HOME/models/llama.cpp/tools/mtmd/test-1.jpeg ./tests/test_qwen35_vision; echo rc=$?
```

Expected: `ds4: Ornith-1.5-35B-A3B runs on single-host Metal only; --vision is not supported`, then `FAIL: engine open`, `rc=1`.

- [ ] **Step 3: Implement**

In `ds4.c`, delete this line from the qwen35moe refusal list (line 74166):

```c
            (opt->vision_path && opt->vision_path[0]) ? "--vision" :
```

In the `--vision` block (lines 74225-74260), admit qwen35moe and bind it like Qwen3.8:

```c
        if (!ds4_model_is_glm53() && !g_ds4_flash_vision_exp && !ds4_model_is_qwen4() &&
            !ds4_model_is_qwen35moe() && DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK41) {
            fprintf(stderr,
                    "ds4: --vision requires GLM-5.3, Qwen3.8-Flash-Next, Ornith-1.5-35B-A3B or the pinned "
                    "DeepSeek V4 Flash Vision-Exp or V4.1 Flash model\n");
```

```c
        } else if (ds4_model_is_qwen4() || ds4_model_is_qwen35moe()) {
            /* Ornith-1.5 ships the Qwen3-VL encoder Qwen3.8 uses (27 blocks, width
             * 1152, merge 2, no deepstack); only the projection width differs, and
             * the check below holds it to the model's embedding width (2048). */
            qwen4_vision_weights_bind(&e->qwen4_vision_weights, &e->vision_model);
            config_expect_u32("vision projection_dim", e->qwen4_vision_weights.n_out, DS4_N_EMBD);
            e->vision_kind = DS4_VISION_QWEN4;
```

The image tokens then resolve from Ornith's vocab at `ds4.c:74621-74624`, which is keyed on `DS4_VISION_QWEN4`.

In `ds4_help.c` and `docs/SERVER.md`, add "Ornith-1.5-35B-A3B" to the models `--vision` accepts. Each file has one sentence listing them; `grep -n "Qwen3.8-Flash-Next" ds4_help.c docs/SERVER.md` finds it.

- [ ] **Step 4: Run it to verify it passes**

Same command as Step 2. Expected: `encode: N rows (H x W)` with H×W = N, then `PASS: ornith vision (vision=1 mtp=0)`, `rc=0`.

- [ ] **Step 5: Commit**

```bash
git add ds4.c ds4_help.c docs/SERVER.md Makefile tests/test_qwen35_vision.c
git commit -m "ornith: --vision loads the Qwen3-VL mmproj (projection 2048) on qwen35moe"
```

---

### Task 2: Image rows and M-RoPE positions in the Ornith graph

**Files:**
- Modify: `ds4_qwen35moe.inc:166-184` (`qwen35_graph_reset`), `:202-213` (`qwen35_graph_stage_inputs`), `:1015-1023` (MTP staging)
- Modify: `ds4.c:79405-79460` (the qwen35 branch of `ds4_session_sync_internal`), plus a new test helper next to `ds4_session_set_test_rewind_point` (`ds4.c:89608`)
- Modify: `ds4.h` (declare the helper next to `ds4_session_set_test_rewind_point`)
- Test: `tests/test_qwen35_vision.c`

**Interfaces:**
- Consumes: Task 1's harness.
- Produces:
  - `bool ds4_session_test_read_pos3(ds4_session *s, int pos, uint32_t out[3]);`
  - `static int32_t qwen35_mrope_delta_at(const ds4_qwen4_gpu_graph *g, uint32_t p);` (in the `.inc`)
  - `g->mrope_delta`, which now holds Ornith's running offset. Task 3 saves it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_qwen35_vision.c` above `main`:

```c
/* Deterministic stand-in for an encoded image: grid gh x gw rows of n_embd. */
static ds4_vision_embedding fake_image(int n_embd, uint32_t gh, uint32_t gw, uint8_t seed) {
    ds4_vision_embedding e = {0};
    e.token_count = gh * gw;
    e.grid_height = gh;
    e.grid_width = gw;
    e.data = malloc((size_t)e.token_count * (size_t)n_embd * sizeof(float));
    if (!e.data) fail("alloc image rows");
    for (size_t i = 0; i < (size_t)e.token_count * (size_t)n_embd; i++)
        e.data[i] = 0.02f * sinf(0.37f * (float)i + (float)seed);
    memset(e.fingerprint, seed, sizeof(e.fingerprint));
    return e;
}

/* text1 [image A 6x8] text2 [image B 5x4] text3: image A straddles the 64-token
 * prefill chunk boundary.  Spans own the image rows. */
static void build_prompt(ds4_engine *e, ds4_tokens *out, ds4_vision_span spans[2]) {
    char err[256];
    const int n = ds4_engine_embd_dim(e);
    ds4_tokenize_text(e, "<|im_start|>user\nLook at the first picture carefully. ", out);
    ds4_vision_embedding a = fake_image(n, 6, 8, 11);
    if (!ds4_prompt_append_vision(e, out, &spans[0], &a, err, sizeof(err))) fail(err);
    ds4_tokens mid = {0};
    ds4_tokenize_text(e, " and now the second one, which is smaller: ", &mid);
    for (int i = 0; i < mid.len; i++) ds4_tokens_push(out, mid.v[i]);
    ds4_tokens_free(&mid);
    ds4_vision_embedding b = fake_image(n, 5, 4, 23);
    if (!ds4_prompt_append_vision(e, out, &spans[1], &b, err, sizeof(err))) fail(err);
    ds4_tokens tail = {0};
    ds4_tokenize_text(e, " What differs?<|im_end|>\n<|im_start|>assistant\n", &tail);
    for (int i = 0; i < tail.len; i++) ds4_tokens_push(out, tail.v[i]);
    ds4_tokens_free(&tail);
}

/* HF get_rope_index, written independently of qwen4_mrope_pos: text takes
 * (n, n, n) and advances n by one; an image takes (n, n+r, n+c) and then
 * advances n by max(rows, cols). */
static void expected_pos3(const ds4_vision_span *sp, size_t n_sp, int len, uint32_t (*out)[3]) {
    uint32_t next = 0;
    size_t k = 0;
    for (int p = 0; p < len;) {
        if (k < n_sp && (uint32_t)p == sp[k].token_start) {
            const uint32_t gh = sp[k].embedding.grid_height, gw = sp[k].embedding.grid_width;
            for (uint32_t r = 0; r < gh; r++)
                for (uint32_t c = 0; c < gw; c++, p++) {
                    out[p][0] = next;
                    out[p][1] = next + r;
                    out[p][2] = next + c;
                }
            next += gh > gw ? gh : gw;
            k++;
        } else {
            out[p][0] = out[p][1] = out[p][2] = next++;
            p++;
        }
    }
}

static void check_pos3(ds4_session *s, const uint32_t (*want)[3], int from, int to, const char *what) {
    for (int p = from; p < to; p++) {
        uint32_t got[3];
        if (!ds4_session_test_read_pos3(s, p, got)) fail("read pos3");
        if (memcmp(got, want[p], sizeof(got)) != 0) {
            fprintf(stderr, "FAIL: %s pos %d got (%u,%u,%u) want (%u,%u,%u)\n", what, p,
                    got[0], got[1], got[2], want[p][0], want[p][1], want[p][2]);
            exit(1);
        }
    }
}

/* positions after a cold multimodal prefill, then 8 decode evals; the same
 * prompt reached by a continuation (prefix up to the middle text, then the
 * rest) gives the same positions and the same last-row argmax */
static void case_positions(ds4_engine *e) {
    char err[256];
    ds4_tokens prompt = {0};
    ds4_vision_span spans[2];
    build_prompt(e, &prompt, spans);
    const int len = prompt.len, extra = 8;
    uint32_t (*want)[3] = malloc((size_t)(len + extra) * sizeof(*want));
    ds4_tokens full = {0};
    for (int i = 0; i < len; i++) ds4_tokens_push(&full, prompt.v[i]);
    for (int i = 0; i < extra; i++) ds4_tokens_push(&full, prompt.v[i % 8]); /* any text ids */
    expected_pos3(spans, 2, len + extra, want);

    ds4_session *x = NULL, *y = NULL;
    if (ds4_session_create(&x, e, CTX) || ds4_session_create(&y, e, CTX)) fail("session create");
    if (ds4_session_sync_multimodal(x, &prompt, spans, 2, err, sizeof(err))) fail(err);
    check_pos3(x, (const uint32_t (*)[3])want, 0, len, "prefill");
    printf("positions: %d prefill rows match get_rope_index\n", len);
    for (int i = 0; i < extra; i++)
        if (ds4_session_eval(x, full.v[len + i], err, sizeof(err))) fail(err);
    check_pos3(x, (const uint32_t (*)[3])want, len, len + extra, "decode");
    printf("decode: %d eval rows continue the offset\n", extra);

    const int cut = (int)(spans[0].token_start + spans[0].embedding.token_count) + 4;
    ds4_tokens head = {0};
    for (int i = 0; i < cut; i++) ds4_tokens_push(&head, prompt.v[i]);
    if (ds4_session_sync_multimodal(y, &head, spans, 1, err, sizeof(err))) fail(err);
    if (ds4_session_sync_multimodal(y, &prompt, spans, 2, err, sizeof(err))) fail(err);
    check_pos3(y, (const uint32_t (*)[3])want, 0, len, "continuation");
    ds4_session_invalidate(x);
    if (ds4_session_sync_multimodal(x, &prompt, spans, 2, err, sizeof(err))) fail(err);
    if (ds4_session_argmax(x) != ds4_session_argmax(y)) fail("continuation changes the next token");
    printf("continuation: offset carried across syncs\n");
    ds4_session_free(x);
    ds4_session_free(y);
    free(want);
    ds4_tokens_free(&head);
    ds4_tokens_free(&full);
    ds4_tokens_free(&prompt);
}

/* text-only prompt: positions are (p, p, p) and the logits checksum must not
 * depend on whether --vision was loaded (the Makefile compares two runs) */
static void case_text_only(ds4_engine *e) {
    char err[256];
    ds4_tokens t = {0};
    ds4_tokenize_text(e, "<|im_start|>user\nName three prime numbers.<|im_end|>\n<|im_start|>assistant\n", &t);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, CTX)) fail("session create");
    if (ds4_session_sync(s, &t, err, sizeof(err))) fail(err);
    for (int p = 0; p < t.len; p++) {
        uint32_t got[3];
        if (!ds4_session_test_read_pos3(s, p, got)) fail("read pos3");
        if (got[0] != (uint32_t)p || got[1] != (uint32_t)p || got[2] != (uint32_t)p) fail("text row is not (p,p,p)");
    }
    const int vocab = ds4_engine_vocab_size(e);
    float *l = malloc((size_t)vocab * sizeof(float));
    if (ds4_session_copy_logits(s, l, vocab) != vocab) fail("copy logits");
    uint32_t h = 2166136261u;
    for (int k = 0; k < vocab; k++) {
        uint32_t bits;
        memcpy(&bits, &l[k], sizeof(bits));
        h = (h ^ bits) * 16777619u;
    }
    printf("TEXT-CHECKSUM %08x argmax %d\n", h, ds4_session_argmax(s));
    free(l);
    ds4_session_free(s);
    ds4_tokens_free(&t);
}
```

Change `main` to run the new cases:

```c
    ds4_engine *e = open_engine(vision, mtp && mtp[0] == '1');
    case_text_only(e);
    if (vision) {
        case_encode(e);
        case_positions(e);
    }
```

Declare the helper in `ds4.h`, next to `ds4_session_set_test_rewind_point`:

```c
/* Test helper (Ornith): the (t, h, w) rope position the graph staged for pos.
 * False for other models or when the read fails. */
bool ds4_session_test_read_pos3(ds4_session *s, int pos, uint32_t out[3]);
```

Add a stub in `ds4.c`, after `ds4_session_set_test_rewind_point`, so that the test fails on positions rather than at link time:

```c
bool ds4_session_test_read_pos3(ds4_session *s, int pos, uint32_t out[3]) {
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_QWEN4_METAL)
    if (!s || !ds4_session_is_qwen35(s) || pos < 0 || !s->qwen4_graph.pos3) return false;
    uint32_t v[4];
    (void)ds4_gpu_synchronize();
    if (!ds4_gpu_tensor_read(s->qwen4_graph.pos3, (uint64_t)pos * 16u, v, sizeof(v))) return false;
    memcpy(out, v, 3 * sizeof(uint32_t));
    return true;
#else
    (void)s; (void)pos; (void)out;
    return false;
#endif
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `make test-qwen35-vision DS4_TEST_MODEL=… DS4_TEST_MMPROJ=… DS4_TEST_IMAGE=…`, with the slot held down and `DS4_LOCK_FILE` set.

Expected:
- The text-only runs print `TEXT-CHECKSUM`.
- The vision run fails with `FAIL: prefill pos <first image row> got (p,p,p) want (…)`.
- Before that failure, `ds4_session_sync_multimodal` may refuse with "vision encoder is not loaded" only if Task 1 is missing.

- [ ] **Step 3: Implement**

In `ds4_qwen35moe.inc`, add the following before `qwen35_graph_stage_inputs`. `qwen4_span_row` and `qwen4_mrope_pos` are defined at `ds4.c:59682-59710`, which is above the include at `ds4.c:62407`.

```c
/* The M-RoPE offset in force at position p: text after an image continues
 * at max(rows, cols) past the image's base, so every image that ends at or
 * before p moves later positions by max(rows, cols) - rows*cols (the update
 * qwen4_mrope_pos applies at an image's last row).  With spans (a prefill)
 * it is recomputed from them, so a chunk, a continuation and the MTP pass
 * over the same positions all agree; without (decode) every image is
 * behind p and the running g->mrope_delta holds it. */
static int32_t qwen35_mrope_delta_at(const ds4_qwen4_gpu_graph *g, uint32_t p) {
    if (!g->vis_span_count) return g->mrope_delta;
    int32_t d = 0;
    for (size_t i = 0; i < g->vis_span_count; i++) {
        const ds4_vision_span *sp = &g->vis_spans[i];
        const uint32_t cnt = sp->embedding.token_count;
        if ((uint64_t)sp->token_start + cnt > p) continue;
        const uint32_t gw = sp->embedding.grid_width > 0 ? sp->embedding.grid_width : cnt;
        const uint32_t gh = sp->embedding.grid_height > 0 ? sp->embedding.grid_height : 1u;
        d -= (int32_t)cnt - (int32_t)(gh > gw ? gh : gw);
    }
    return d;
}
```

Replace `qwen35_graph_stage_inputs`:

```c
/* Rows: the token's embedding (dequantized on the host from Q8_0), or at an
 * image position the image's row.  Rope positions (t, h, w): (p, p, p) +
 * the running offset for text, the image grid inside an image. */
static bool qwen35_graph_stage_inputs(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                      const int *tokens, uint32_t T) {
    const uint32_t E = DS4_N_EMBD;
    const ds4_vision_span *spans = g->vis_spans;
    const size_t n_spans = g->vis_span_count;
    if (n_spans) g->mrope_delta = qwen35_mrope_delta_at(g, g->pos);
    for (uint32_t t = 0; t < T; t++) {
        float *dst = g->host_row + (uint64_t)t * E;
        const float *img = n_spans ? qwen4_span_row(spans, n_spans, g->pos + t) : NULL;
        if (img) memcpy(dst, img, E * sizeof(float));
        else qwen4_ref_row(m, w->token_embd, (uint64_t)tokens[t], dst);
        uint32_t *p4 = g->host_pos3 + (uint64_t)t * 4u;
        qwen4_mrope_pos(spans, n_spans, g->pos + t, &g->mrope_delta, p4);
        p4[3] = 0;
    }
    return ds4_gpu_tensor_write(g->R, 0, g->host_row, (uint64_t)T * E * sizeof(float)) &&
           ds4_gpu_tensor_write(g->pos3, (uint64_t)g->pos * 16u, g->host_pos3, (uint64_t)T * 16u);
}
```

In `qwen35_graph_mtp` (`ds4_qwen35moe.inc:1015-1023`), replace the staging loop body:

```c
        int32_t d = qwen35_mrope_delta_at(g, p);
        for (uint32_t t = 0; t < n; t++) {
            float *dst = g->host_row + (uint64_t)t * E;
            const float *img = g->vis_span_count ?
                qwen4_span_row(g->vis_spans, g->vis_span_count, p + t) : NULL;
            if (img) memcpy(dst, img, E * sizeof(float));
            else qwen4_ref_row(m, w->token_embd, (uint64_t)tokens[done + t], dst);
            uint32_t *p4 = g->host_pos3 + (uint64_t)t * 4u;
            qwen4_mrope_pos(g->vis_spans, g->vis_span_count, p + t, &d, p4);
            p4[3] = 0;
        }
```

In `qwen35_graph_reset`, after `g->mtp_h_rows = 0;`, add:

```c
    g->mrope_delta = 0;
```

In `ds4.c`'s qwen35 sync branch, hand the spans to the graph before the prefill loop, as the qwen4 branch does (`ds4.c:79493`). Before `for (int i = start; i < prompt->len;) {` add:

```c
        g->vis_spans = s->sync_images;          /* NULL for a text sync */
        g->vis_span_count = s->sync_image_count;
```

The loop has three early returns (interrupted, prefill failed, MTP failed), so clear the pointer once, in the caller that owns the spans. Otherwise a later decode would read freed image rows. In `ds4_session_sync_multimodal` (`ds4.c:79362-79373`), next to the existing `s->graph.prefill_vision_spans = NULL;` reset after `ds4_session_sync` returns, add:

```c
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_QWEN4_METAL)
    s->qwen4_graph.vis_spans = NULL;
    s->qwen4_graph.vis_span_count = 0;
#endif
```

Replace the stub `ds4_session_test_read_pos3` body only if a compile error says otherwise. It is already the final implementation.

- [ ] **Step 4: Run it to verify it passes**

Same command. Expected:
- `cmp` is silent (text-only checksums equal with and without `--vision`).
- Output lines: `positions: N prefill rows match get_rope_index`, `decode: 8 eval rows continue the offset`, `continuation: offset carried across syncs`.
- `PASS` for both the plain and the MTP run. The MTP run's `positions` line also proves the MTP staging writes, because MTP catch-up rewrites the same `pos3` rows after the trunk.

Then run the existing Ornith model tests, which must be unchanged:
- `make test-qwen35-rewind-point DS4_TEST_MODEL=…`
- `./tests/test_qwen35_mtp <model>`, after `make tests/test_qwen35_mtp`

Expected: `PASS` for both.

- [ ] **Step 5: Commit**

```bash
git add ds4_qwen35moe.inc ds4.c ds4.h tests/test_qwen35_vision.c
git commit -m "ornith: image rows and M-RoPE positions in trunk and MTP staging (offset recomputed from spans)"
```

---

### Task 3: MTP losslessness after images; rewind point and payload load carry the offset

**Files:**
- Modify: `ds4.c:63455-63462` (session fields next to `rewind_mtp_pos`), `ds4.c:89530-89548` (restore), `ds4.c:89580-89600` (mark)
- Modify: `ds4_qwen35moe.inc` or `ds4.c:66226` (`qwen35_session_load_payload`: zero the offset)
- Test: `tests/test_qwen35_vision.c`

**Interfaces:**
- Consumes: `g->mrope_delta` (Task 2).
- Produces: session field `int32_t rewind_mrope_delta;` and `static void ds4_session_trim_vision_identities(ds4_session *s, int pos);`. Task 5's server path relies on the trimming.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_qwen35_vision.c`:

```c
/* speculative decoding after an image prompt commits exactly the plain
 * argmax sequence (MTP engine only) */
static void case_mtp(ds4_engine *e) {
    char err[256];
    ds4_tokens prompt = {0};
    ds4_vision_span spans[2];
    build_prompt(e, &prompt, spans);
    ds4_session *spec = NULL, *plain = NULL;
    if (ds4_session_create(&spec, e, CTX) || ds4_session_create(&plain, e, CTX)) fail("session create");
    if (ds4_session_sync_multimodal(spec, &prompt, spans, 2, err, sizeof(err))) fail(err);
    if (ds4_session_sync_multimodal(plain, &prompt, spans, 2, err, sizeof(err))) fail(err);
    int committed = 0, cycles = 0, accepted = 0;
    while (committed < 96) {
        const int first = ds4_session_argmax(spec);
        int acc[17];
        const int n = ds4_session_eval_speculative_argmax(spec, first, 16, -1, acc, 17, err, sizeof(err));
        if (n < 1) fail(err[0] ? err : "speculative step");
        for (int i = 0; i < n; i++) {
            if (ds4_session_argmax(plain) != acc[i]) fail("speculative token differs from plain argmax");
            if (ds4_session_eval(plain, acc[i], err, sizeof(err))) fail(err);
        }
        committed += n;
        cycles++;
        accepted += n - 1;
    }
    printf("mtp: %d tokens in %d cycles, %d drafts accepted, identical to plain\n", committed, cycles, accepted);
    ds4_session_free(spec);
    ds4_session_free(plain);
    ds4_tokens_free(&prompt);
}

/* mark after image A, continue with image B (the offset moves), rewind to the
 * mark: the offset and the image identities come back, and the next rows
 * match a session that never saw image B */
static void case_rewind(ds4_engine *e) {
    char err[256];
    ds4_tokens prompt = {0};
    ds4_vision_span spans[2];
    build_prompt(e, &prompt, spans);
    const int mark = (int)(spans[0].token_start + spans[0].embedding.token_count) + 4;
    ds4_tokens head = {0};
    for (int i = 0; i < mark; i++) ds4_tokens_push(&head, prompt.v[i]);
    ds4_session *x = NULL, *y = NULL;
    if (ds4_session_create(&x, e, CTX) || ds4_session_create(&y, e, CTX)) fail("session create");
    if (ds4_session_sync_multimodal(x, &head, spans, 1, err, sizeof(err))) fail(err);
    if (!ds4_session_mark_rewind_point(x)) fail("mark after an image");
    if (ds4_session_sync_multimodal(x, &prompt, spans, 2, err, sizeof(err))) fail(err);
    ds4_session_rewind(x, mark);
    if (!ds4_session_checkpoint_valid(x) || ds4_session_pos(x) != mark) fail("rewind to the point");
    if (!ds4_session_vision_state_matches(x, spans, 1)) fail("rewind kept image B's identity");
    if (ds4_session_sync_multimodal(y, &head, spans, 1, err, sizeof(err))) fail(err);
    const int vocab = ds4_engine_vocab_size(e);
    float *lx = malloc((size_t)vocab * sizeof(float)), *ly = malloc((size_t)vocab * sizeof(float));
    float worst = 0.0f;
    for (int i = 0; i < 16; i++) {
        const int tok = prompt.v[i % 8];
        if (ds4_session_eval(x, tok, err, sizeof(err)) || ds4_session_eval(y, tok, err, sizeof(err))) fail(err);
        ds4_session_copy_logits(x, lx, vocab);
        ds4_session_copy_logits(y, ly, vocab);
        if (ds4_session_argmax(x) != ds4_session_argmax(y)) fail("rewind: argmax differs");
        for (int k = 0; k < vocab; k++) if (fabsf(lx[k] - ly[k]) > worst) worst = fabsf(lx[k] - ly[k]);
    }
    if (!(worst <= 1e-2f)) fail("rewind: logits differ");
    uint32_t px[3], py[3];
    if (!ds4_session_test_read_pos3(x, mark + 15, px) || !ds4_session_test_read_pos3(y, mark + 15, py) ||
        memcmp(px, py, sizeof(px))) fail("rewind: offset not restored");
    printf("rewind: offset and identities restored, max |dlogit| %.6g\n", worst);
    free(lx);
    free(ly);
    ds4_session_free(x);
    ds4_session_free(y);
    ds4_tokens_free(&head);
    ds4_tokens_free(&prompt);
}

/* a session that held images, then a text-only payload load: offset zero */
static void case_payload(ds4_engine *e) {
    char err[256];
    ds4_tokens prompt = {0}, text = {0};
    ds4_vision_span spans[2];
    build_prompt(e, &prompt, spans);
    ds4_tokenize_text(e, "<|im_start|>user\nCount to five.<|im_end|>\n<|im_start|>assistant\n", &text);
    ds4_session *x = NULL, *y = NULL;
    if (ds4_session_create(&x, e, CTX) || ds4_session_create(&y, e, CTX)) fail("session create");
    if (ds4_session_sync_multimodal(x, &prompt, spans, 2, err, sizeof(err))) fail(err);
    if (ds4_session_sync(y, &text, err, sizeof(err))) fail(err);
    ds4_session_snapshot snap = {0};
    if (ds4_session_save_snapshot(y, &snap, err, sizeof(err))) fail(err);
    if (ds4_session_load_snapshot(x, &snap, err, sizeof(err))) fail(err);
    if (ds4_session_eval(x, text.v[0], err, sizeof(err))) fail(err);
    uint32_t got[3];
    if (!ds4_session_test_read_pos3(x, text.len, got)) fail("read pos3");
    if (got[0] != (uint32_t)text.len) fail("payload load kept the image offset");
    printf("payload: offset reset on a text-only load\n");
    ds4_session_snapshot_free(&snap);
    ds4_session_free(x);
    ds4_session_free(y);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&text);
}
```

In `main`, inside `if (vision)`:

```c
        case_rewind(e);
        case_payload(e);
        if (mtp && mtp[0] == '1') case_mtp(e);
```

- [ ] **Step 2: Run it to verify it fails**

Run `make test-qwen35-vision …` with the slot held down. Expected failure:
- `FAIL: rewind kept image B's identity`, or the `fail("rewind to the point")`;
- with the identity check commented out, `FAIL: rewind: offset not restored` / `argmax differs`.

`case_payload` fails with "payload load kept the image offset". `case_mtp` is expected to pass already (Task 2 staged MTP rows). If it does, it pins losslessness and is not a RED; record that in the ledger.

- [ ] **Step 3: Implement**

Add the session field in `ds4.c` next to `rewind_mtp_pos` (line 63459):

```c
    int32_t rewind_mrope_delta;   /* Ornith M-RoPE offset at rewind_pos */
```

Add a helper before `ds4_session_rewind`:

```c
/* After a rewind to pos, image identities past pos describe rows that no
 * longer exist; images are stored in prompt order. */
static void ds4_session_trim_vision_identities(ds4_session *s, int pos) {
    size_t keep = 0;
    while (keep < s->checkpoint_image_count &&
           (uint64_t)s->checkpoint_images[keep].token_start +
           s->checkpoint_images[keep].token_count <= (uint64_t)pos) keep++;
    s->checkpoint_image_count = keep;
}
```

In the rewind-point restore branch (`ds4.c:89540-89547`), inside `if (ok) {`:

```c
            g->mrope_delta = s->rewind_mrope_delta;
            ds4_session_trim_vision_identities(s, pos);
```

In `ds4_session_mark_rewind_point`, next to `s->rewind_mtp_pos = g->mtp_pos;`:

```c
    s->rewind_mrope_delta = g->mrope_delta;
```

In `qwen35_session_load_payload` (`ds4.c:66226`), after the `if (!s->qwen35_graph_ready)` check:

```c
    g->mrope_delta = 0;   /* payloads never hold image-conditioned state */
```

- [ ] **Step 4: Run it to verify it passes**

Same command. Expected:
- new lines `rewind: offset and identities restored, max |dlogit| …` (≤ 1e-2) and `payload: offset reset on a text-only load`;
- in the MTP run, `mtp: 96+ tokens …, identical to plain`.

Then rerun `make test-qwen35-rewind-point …`. Expected: `PASS` for both runs.

- [ ] **Step 5: Commit**

```bash
git add ds4.c ds4_qwen35moe.inc tests/test_qwen35_vision.c
git commit -m "ornith: rewind point and payload load carry the M-RoPE offset; rewind trims image identities"
```

---

### Task 4: Image markers survive the Ornith renderer (pin)

**Files:**
- Test: `ds4_server.c` (unit test next to `test_visible_image_key`, line ~25360; register next to `test_rewind_point_option();`, line 26376)

**Interfaces:**
- Consumes: `render_ornith_chat_prompt_text(const chat_msgs *, const char *tool_schemas, const tool_schema_orders *, ds4_think_mode, const chat_template_opts *)`.

- [ ] **Step 1: Write the test**

```c
static void test_ornith_render_keeps_image_markers(void) {
    chat_msg m[3] = {0};
    m[0].role = (char *)"user";
    m[0].content = (char *)"look at this \036DS4_IMAGE_aa\037 please";
    m[1].role = (char *)"assistant";
    m[1].content = (char *)"Reading the file.";
    m[2].role = (char *)"tool";
    m[2].tool_call_id = (char *)"t1";
    m[2].content = (char *)"\036DS4_IMAGE_bb\037 image read";
    chat_msgs msgs = {.v = m, .len = 3, .cap = 3};
    char *text = render_ornith_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_NONE, &CHAT_TEMPLATE_DEFAULTS);
    TEST_ASSERT(text != NULL);
    const char *a = strstr(text, "\036DS4_IMAGE_aa\037");
    const char *b = strstr(text, "\036DS4_IMAGE_bb\037");
    TEST_ASSERT(a && b && a < b);
    const char *tr = strstr(text, "<tool_response>");
    TEST_ASSERT(tr && tr < b && strstr(b, "</tool_response>"));
    free(text);
}
```

Register it after `test_rewind_point_option();`.

- [ ] **Step 2: Run it**

Run: `make -j8 ds4_test > <scratch>/v4.log 2>&1 && ./ds4_test --server > <scratch>/v4t.log 2>&1; echo rc=$?`

Expected: `rc=0`. The Ornith renderer copies content through `append_trimmed_text`, and the marker bytes 0x1E/0x1F are not `isspace`. A pass pins that behaviour, and the ledger records "pin, passed first run".

If it fails, the renderer drops or splits the markers. Make the user and tool paths in `append_ornith_conversation` copy content byte-for-byte between the markers (they must not trim inside the content), then rerun.

- [ ] **Step 3: Commit**

```bash
git add ds4_server.c
git commit -m "test: Ornith renderer keeps image markers in user content and tool responses"
```

---

### Task 5: Rewind point for image conversations (server)

**Files:**
- Modify: `ds4_server.c:10875-10882` (`rewind_point` struct), `:11372-11392` (clear/remember), `:12880-12893` (probe tier), `:12530-12550` (`rewind_point_eval_to`), `:15120-15152` (mark and hit)
- Test: `ds4_server.c` (next to `test_rewind_point_reuse_tier`, line 20169; register in the list at 26376-26380)

**Interfaces:**
- Consumes: `visible_prompt_key(const request *, const char *, visible_image_key *)` (`ds4_server.c:10802`) and the Task 3 engine trimming.
- Produces:
  - `static void rewind_point_remember(server *s, server_slot *slot, const request *req, size_t text_len, int live_tokens);` (the signature changes: it now takes the request);
  - `static bool rewind_point_images_match(const visible_image_key *incoming, const visible_image_key *old, size_t text_len);`
  - `slot->rewind_point.images`.

- [ ] **Step 1: Write the failing tests**

```c
static void test_rewind_point_images_match(void) {
    visible_image_key old = {.count = 1, .offsets = {5}};
    visible_image_key same = {.count = 1, .offsets = {5}};
    visible_image_key moved = {.count = 1, .offsets = {6}};
    visible_image_key extra = {.count = 2, .offsets = {5, 40}};
    visible_image_key none = {0};
    TEST_ASSERT(rewind_point_images_match(&same, &old, 30));
    TEST_ASSERT(!rewind_point_images_match(&moved, &old, 30));
    TEST_ASSERT(!rewind_point_images_match(&extra, &old, 30));   /* an image after the point */
    TEST_ASSERT(!rewind_point_images_match(&none, &old, 30));
    TEST_ASSERT(rewind_point_images_match(&none, &none, 30));
}

static void test_rewind_point_reuse_tier_images(void) {
    server s = {0};
    int live[300];
    for (int i = 0; i < 300; i++) live[i] = 7000 + i;
    server_slot slot = {0};
    slot.session = ds4_session_new_test_checkpoint(live, 300);
    ds4_vision_span img = {.token_start = 20};
    img.embedding.token_count = 16;
    memset(img.embedding.fingerprint, 7, sizeof(img.embedding.fingerprint));
    ds4_session_set_test_images(slot.session, &img, 1);
    ds4_session_set_test_rewind_point(slot.session, 200);
    char markers[1][SERVER_IMAGE_MARKER_BYTES] = {"\036DS4_IMAGE_aa\037"};
    request first = {.kind = REQ_CHAT, .image_count = 1, .image_markers = markers,
                     .prompt_text = (char *)"<sys>u:\036DS4_IMAGE_aa\037 see<tool>"};
    first.images = &img;
    rewind_point_remember(&s, &slot, &first, strlen(first.prompt_text), 200);
    TEST_ASSERT(slot.rewind_point.valid && slot.rewind_point.images.count == 1);

    /* the re-send carries a fresh nonce: normalized, it matches */
    char fresh[1][SERVER_IMAGE_MARKER_BYTES] = {"\036DS4_IMAGE_zz\037"};
    job j = {0};
    ds4_tokens_push(&j.req.prompt, 1);
    j.req.kind = REQ_CHAT;
    j.req.image_count = 1;
    j.req.image_markers = fresh;
    j.req.images = &img;
    j.req.prompt_text = (char *)"<sys>u:\036DS4_IMAGE_zz\037 see<tool><user>again";
    slot_reuse pr = slot_probe_reuse_locked(&s, &slot, &j.req);
    TEST_ASSERT(pr.kind == REUSE_REWIND_POINT && pr.reuse_tokens == 200);

    /* the image moved one byte: miss */
    j.req.prompt_text = (char *)"<sys>u: \036DS4_IMAGE_zz\037see<tool><user>again";
    TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &j.req).kind != REUSE_REWIND_POINT);

    /* a different picture (fingerprint): the probe's vision gate refuses */
    ds4_vision_span other = img;
    memset(other.embedding.fingerprint, 9, sizeof(other.embedding.fingerprint));
    j.req.images = &other;
    j.req.prompt_text = (char *)"<sys>u:\036DS4_IMAGE_zz\037 see<tool><user>again";
    TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &j.req).kind != REUSE_REWIND_POINT);

    rewind_point_clear(&s, &slot);
    ds4_session_free_test_checkpoint(slot.session);
    ds4_tokens_free(&j.req.prompt);
}
```

Register both after `test_rewind_point_reuse_tier();`. Update `test_rewind_point_reuse_tier` (line 20169), which assigns `slot.rewind_point.text` directly. Keep that: a text-only key is its own normalized form. It needs no change unless the struct field names change.

- [ ] **Step 2: Run them to verify they fail**

Run: `make -j8 ds4_test 2>&1 | grep -E "error" | head -5`

Expected: compile errors, because `rewind_point_images_match` is undeclared and `rewind_point_remember` is called with a `request *`.

- [ ] **Step 3: Implement**

Struct (`ds4_server.c:10878-10882`):

```c
    struct {
        bool valid;
        int live_tokens;
        char *text;                 /* marker-normalized (visible_prompt_key) */
        size_t text_len;
        visible_image_key images;   /* image offsets inside text */
    } rewind_point;
```

Remember (replace `rewind_point_remember`):

```c
static void rewind_point_remember(server *s, server_slot *slot, const request *req,
                                  size_t text_len, int live_tokens) {
    visible_image_key images;
    char *key = visible_prompt_key(req, req->prompt_text, &images);
    if (!key || text_len > strlen(key)) {
        free(key);
        rewind_point_clear(s, slot);
        return;
    }
    key[text_len] = '\0';           /* normalization keeps byte offsets */
    while (images.count && images.offsets[images.count - 1] >= text_len) images.count--;
    pthread_mutex_lock(&s->tool_mu);
    free(slot->rewind_point.text);
    slot->rewind_point.text = key;
    slot->rewind_point.text_len = text_len;
    slot->rewind_point.images = images;
    slot->rewind_point.live_tokens = live_tokens;
    slot->rewind_point.valid = true;
    pthread_mutex_unlock(&s->tool_mu);
    server_log(DS4_LOG_KVCACHE, "ds4-server: rewind point remembered pos=%d text=%zu images=%zu",
               live_tokens, text_len, images.count);
}

/* Same images at the same offsets up to the point and none after it: what
 * follows the point is evaluated as text.  Pixels are checked by the probe's
 * fingerprint gate and again when build_live_prompt_suffix rebases spans. */
static bool rewind_point_images_match(const visible_image_key *incoming,
                                      const visible_image_key *old, size_t text_len) {
    if (incoming->count != old->count) return false;
    for (size_t i = 0; i < old->count; i++)
        if (incoming->offsets[i] != old->offsets[i] || incoming->offsets[i] >= text_len) return false;
    return true;
}
```

Probe tier (`ds4_server.c:12883-12893`): replace the condition block:

```c
    if (ptext && req->kind == REQ_CHAT && slot->rewind_point.valid && slot->rewind_point.text &&
        slot->rewind_point.text_len < plen &&
        slot->rewind_point.live_tokens <= live_pos &&
        ds4_session_rewind_point_pos(slot->session) == slot->rewind_point.live_tokens) {
        visible_image_key vimages;
        char *vkey = visible_prompt_key(req, ptext, &vimages);
        const bool hit = vkey &&
            rewind_point_images_match(&vimages, &slot->rewind_point.images, slot->rewind_point.text_len) &&
            byte_prefix_match(vkey, plen, slot->rewind_point.text, slot->rewind_point.text_len);
        free(vkey);
        if (hit) {
            pr.kind = REUSE_REWIND_POINT;
            pr.reuse_tokens = slot->rewind_point.live_tokens;
            pr.suffix_off = slot->rewind_point.text_len;
            return pr;
        }
    }
```

`rewind_point_eval_to` (`ds4_server.c:~12535`): a gap that holds image rows cannot go through token evals, so it returns 0 and the caller prefills it. Add a `const request *r` parameter: `static int rewind_point_eval_to(server *s, server_slot *slot, const request *r, const ds4_tokens *prompt, int target, char *err, size_t errlen)`. Every call site is in `generate_job` and passes `&j->req`. After the `pos`/`target` check, add:

```c
    for (size_t i = 0; r && i < r->image_count; i++) {
        const uint64_t a = r->images[i].token_start, b = a + r->images[i].embedding.token_count;
        if (a < (uint64_t)target && (uint64_t)pos < b) return 0;   /* an image in the gap: prefill it */
    }
```

Mark and hit (`ds4_server.c:15120-15152`):
- `rewind_cut` drops `!multimodal`, but refuses a cut that has an image after it:

  ```c
    const int rewind_cut = (prompt_sync_rc == 0 &&
                            (!multimodal || rewind_point_images_before(&j->req, prompt_for_sync->len))) ?
        rewind_point_cut(s, &j->req, prompt_for_sync, cached, &rewind_text_len) : 0;
  ```

  with the following helper. The tail after the cut is the generation prompt, so `rewind_point_cut` returns a cut ≤ `prompt->len - 1`; images must end at or before it. Re-check after the cut is known:

  ```c
  static bool rewind_point_images_before(const request *r, int limit) {
      for (size_t i = 0; i < r->image_count; i++)
          if ((uint64_t)r->images[i].token_start + r->images[i].embedding.token_count > (uint64_t)limit)
              return false;
      return true;
  }
  ```

  and immediately after computing `rewind_cut`:

  ```c
    if (rewind_cut > 0 && multimodal && !rewind_point_images_before(&j->req, rewind_cut)) rewind_cut = 0;
  ```

  (`rewind_cut` becomes non-`const`.)
- The prefix sync inside the mark path becomes multimodal when the request has images:

  ```c
            prompt_sync_rc = multimodal ?
                server_session_sync_multimodal(s, slot, &prefix, j->req.images, j->req.image_count,
                                               err, sizeof(err)) :
                server_session_sync(s, slot, &prefix, err, sizeof(err));
  ```

- The call becomes `rewind_point_remember(s, slot, &j->req, rewind_text_len, rewind_cut);`.
- The hit branch drops `!multimodal`: `} else if (prompt_sync_rc == 0 && live_materialized && reuse.kind == REUSE_REWIND_POINT) {`.
- All three `rewind_point_eval_to(...)` calls pass `&j->req`.
- Field-frequency evidence (spec §7.4): in `generate_job`, right after `multimodal` is computed, log the image count once per image request. Task 6's C1 asserts this line.

  ```c
    if (j->req.image_count)
        server_log(DS4_LOG_KVCACHE, "ds4-server: request images=%zu prompt=%d",
                   j->req.image_count, j->req.prompt.len);
  ```

- [ ] **Step 4: Run them to verify they pass**

Run: `make -j8 ds4 ds4-server ds4_test > <scratch>/v5b.log 2>&1 && ./ds4_test --server > <scratch>/v5t.log 2>&1; echo rc=$?; grep -c PASS <scratch>/v5t.log`

Expected: `rc=0`. The existing rewind-point tests (`test_rewind_point_*`) still pass.

- [ ] **Step 5: Commit**

```bash
git add ds4_server.c
git commit -m "server: rewind point for image conversations (normalized key + image offsets, multimodal mark, image-free gap)"
```

---

### Task 6: End-to-end acceptance on the production argv

**Files:**
- Create: `speed-bench/ornith-vision/make_fixtures.py`, `speed-bench/ornith-vision/fixtures/` (newspaper, code screenshot, Vietnamese text), `speed-bench/ornith-vision/e2e_vision.py`, `speed-bench/ornith-vision/E2E.md`

**Interfaces:**
- Consumes: the registry row's `process_command` for `…-vision-mtplx-512K` (read with `json` from `~/.local/ai-gateway/runtime-registry.json`, as `speed-bench/rewind-point/bench_resend.py` does).

- [ ] **Step 1: Make the fixtures**

`make_fixtures.py` runs on `/usr/bin/python3`, which has PIL 11.3.0.
- It copies `~/models/llama.cpp/tools/mtmd/test-1.jpeg` to `fixtures/newspaper.jpg`.
- It draws two PNGs. Fonts: `/System/Library/Fonts/Menlo.ttc` and `/System/Library/Fonts/Supplemental/Arial Unicode.ttf`.

```python
#!/usr/bin/env python3
"""Fixtures for the Ornith vision E2E and quality runs (deterministic)."""
import os, shutil
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")
os.makedirs(OUT, exist_ok=True)
shutil.copyfile(os.path.expanduser("~/models/llama.cpp/tools/mtmd/test-1.jpeg"),
                os.path.join(OUT, "newspaper.jpg"))

CODE = '''def parse_invoice_total(lines):
    total = 0
    for line in lines:
        if line.startswith("INV-2041"):
            total += int(line.split(":")[1])
    return total
'''
mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
img = Image.new("RGB", (900, 260), (30, 30, 30))
ImageDraw.Draw(img).multiline_text((20, 20), CODE, font=mono, fill=(220, 220, 220), spacing=8)
img.save(os.path.join(OUT, "code.png"))

VI = "Hóa đơn số 2041\nNgày 10 tháng 10 năm 2026\nTổng cộng: 1.250.000 đồng"
uni = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 34)
img = Image.new("RGB", (760, 220), (255, 255, 255))
ImageDraw.Draw(img).multiline_text((24, 24), VI, font=uni, fill=(0, 0, 0), spacing=14)
img.save(os.path.join(OUT, "vietnamese.png"))
print("fixtures written to", OUT)
```

Run: `/usr/bin/python3 speed-bench/ornith-vision/make_fixtures.py`. Expected: `fixtures written to …`, plus three files.

- [ ] **Step 2: Write `e2e_vision.py`**

The script:
1. Reads the 512K row's `process_command`. It replaces `--port 18089` with `--port 18189` and `--kv-disk-dir …` with a temp dir, sets `--rewind-point-min-tokens 512` (so the short E2E prompts mark), and appends `--vision <mmproj>`.
2. Launches it with `DS4_LOCK_FILE=<tmp>/ds4.lock`, logging to `<tmp>/server.log`, and waits for `GET /v1/models`.
3. Sends OpenAI chat requests (`temperature: 0`, `max_tokens: 200`, `chat_template_kwargs: {"enable_thinking": false, "preserve_thinking": true}`), with images as `data:image/jpeg;base64,…` / `image/png`.
4. Runs these checks and records PASS/FAIL per check:
   - **C1 newspaper:** "Describe this front page: newspaper, date, main headline." The reply (lower-cased) contains `new york times`, `1969` and `men walk on moon`. The server log has `request images=1`.
   - **C2 code:** "Transcribe the function name and the string literal in this screenshot." It contains `parse_invoice_total` and `inv-2041`.
   - **C3 multi-turn:** C1's messages, plus C1's reply as an assistant turn, plus a user turn "Answer with only the main headline." The reply contains `men walk on moon`. The response's `usage.prompt_tokens_details.cached_tokens` is at least C1's `usage.prompt_tokens`, so the image turn was reused live rather than prefilled again.
   - **C4 re-send:** C3's request sent again. The log has `rewind point hit`, and the reply equals C3's.
   - **C5 compaction:** C3's messages plus a user turn "Summarize the conversation in one sentence." The log has `rewind point hit`.
   - **C6 MTP:** relaunch without `--mtp`, rerun C1, and the reply text equals the `--mtp` reply.
   - **C7 Anthropic tool image:** `POST /v1/messages` with an assistant `tool_use` (`Read`) followed by a user `tool_result` whose content is the code PNG, then "What function is in the image the tool returned?". The reply contains `parse_invoice_total`.
5. Writes `speed-bench/ornith-vision/e2e-<date>.json` with each check, its reply length, `ttft_s`, and log-line counts (never prompt text), and SIGTERMs the server.

Base it on `speed-bench/rewind-point/bench_resend.py` (registry read, launch, wait, request, log scan). Copy those helpers rather than importing, because the bench directory is not a package.

- [ ] **Step 3: Run it**

Hold the slot down (quiet window, lock holder, SIGTERM), then run:

```bash
python3 speed-bench/ornith-vision/e2e_vision.py --bin ~/orca/workspaces/ds4-metal/ornith-vision \
  --mmproj ~/orca/workspaces/ds4-metal-data/gguf/ornith-mmproj/mmproj-Ornith-1.5-35B-A3B-f16.gguf
```

Release the holder afterwards.

Expected: `C1..C7 PASS`. Any FAIL goes to superpowers:systematic-debugging; never loosen a check.

- [ ] **Step 4: Write `E2E.md`** with the table of checks, TTFTs and the server-log evidence counts. Then commit:

```bash
git add speed-bench/ornith-vision/make_fixtures.py speed-bench/ornith-vision/fixtures speed-bench/ornith-vision/e2e_vision.py speed-bench/ornith-vision/E2E.md speed-bench/ornith-vision/e2e-*.json
git commit -m "speed-bench: Ornith vision end-to-end acceptance on the production argv (C1-C7)"
```

---

### Task 7: Gateway sends Ornith's images natively, with the sidecar for what ds4 cannot take

Work in an AI-Gateway-MLX worktree: `git worktree add .claude/worktrees/ornith-vision -b dongnh311/ornith-vision develop`.

**Files:**
- Modify: `gateway/vision.py` (after `find_image_parts`, line 167)
- Modify: `gateway/providers/ds4.py:54-56`
- Modify: `gateway/server.py:1637` (Phase-1 selection), `:1768-1774` (comment)
- Create: `tests/test-suite-vision-native.py`
- Modify: `tests/run-success-gate.py` (register `H.vision_native` next to `H.vision_edge_deadline`, line 1650)

**Interfaces:**
- Produces:
  - `vision.native_split(obj, limit, types) -> [(parts, index, part)]`;
  - `DS4Provider.native_image_limit = 16`;
  - `DS4Provider.native_image_types = ("image/png", "image/jpeg", "image/jpg")`.

- [ ] **Step 1: Write the failing test**

`tests/test-suite-vision-native.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hermetic: a native-vision ds4 route gets the images ds4 can take (the first 16 PNG/JPEG of the
conversation, by arrival), and every other image part goes to the describe sidecar.

WHY. ds4-server refuses a request with more than 16 images and accepts only PNG and JPEG data
(ds4_server.c: "at most 16", server_image_media_type). The cut is by ARRIVAL, never by recency:
turning an old image into text would change the prompt prefix and force a full re-prefill.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "gateway"))

RESULTS = []


def case(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("%s %-72s %s" % ("✅" if ok else "❌", name, note))


for mod in ("vision", "config"):
    sys.modules.pop(mod, None)
import vision                                                    # noqa: E402
from providers import ds4 as ds4prov                             # noqa: E402


def img(kind, n):
    data = "QUJD%04d" % n
    if kind == "png":
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}
    if kind == "jpeg":
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}
    if kind == "webp":
        return {"type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": data}}
    return {"type": "input_audio", "input_audio": {"data": data}}


def convo(kinds):
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": "t%d" % i}, img(k, i)]}
                         for i, k in enumerate(kinds)]}


TYPES = ("image/png", "image/jpeg", "image/jpg")

obj = convo(["png"] * 18)
out = vision.native_split(obj, 16, TYPES)
case("18 PNG: the last 2 go to the sidecar", [p[2] for p in out] ==
     [obj["messages"][16]["content"][1], obj["messages"][17]["content"][1]])

obj = convo(["png", "webp", "jpeg", "audio"])
out = vision.native_split(obj, 16, TYPES)
case("webp and audio go to the sidecar, png/jpeg stay native",
     [p[2].get("type") for p in out] == ["image", "input_audio"] and
     out[0][2]["source"]["media_type"] == "image/webp")

obj = convo(["webp"] + ["png"] * 16)
out = vision.native_split(obj, 16, TYPES)
case("a rejected type does not use up one of the 16", len(out) == 1)

grown = convo(["png"] * 17)
first = vision.native_split(convo(["png"] * 17), 16, TYPES)
grown["messages"].append({"role": "user", "content": [img("png", 99)]})
second = vision.native_split(grown, 16, TYPES)
case("growing the conversation never changes which OLD images are native",
     [id(p[2]) for p in second][:1] != [] and len(second) == len(first) + 1)

case("limit 0 sends nothing to the sidecar", vision.native_split(convo(["png"] * 20), 0, TYPES) == [])

case("DS4Provider declares ds4's limits",
     ds4prov.DS4Provider.native_image_limit == 16 and
     set(ds4prov.DS4Provider.native_image_types) == set(TYPES))

print("\n%d/%d" % (sum(RESULTS), len(RESULTS)))
sys.exit(0 if all(RESULTS) else 1)
```

Register in `tests/run-success-gate.py` after the `H.vision_edge_deadline` tuple:

```python
        ("H.vision_native", "Stub suite: a native-vision ds4 route keeps the first 16 PNG/JPEG "
                            "images of a conversation (by arrival, so the prompt prefix never "
                            "changes) and sends every other image part to the describe sidecar "
                            "(ds4-server: at most 16 images, PNG/JPEG only)",
         lambda: [py, "tests/test-suite-vision-native.py"], []),
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 tests/test-suite-vision-native.py; echo rc=$?`

Expected: `AttributeError: module 'vision' has no attribute 'native_split'`, `rc=1`.

- [ ] **Step 3: Implement**

`gateway/vision.py`, after `find_image_parts`:

```python
def native_split(obj, limit, types):
    """Content parts a native-vision backend must NOT receive, in arrival order: every non-image
    part, every image whose media type it does not accept, and every accepted image after the
    first `limit` of the conversation. The caller describes these with the sidecar like on a
    text-only route. limit <= 0 means the backend has no native image path: nothing is split off.

    The cut is by ARRIVAL, never by recency: a growing conversation keeps the same images native,
    so the prompt prefix stays byte-stable (turning an OLD image into text forces a full
    re-prefill on a long session)."""
    if not limit or limit <= 0:
        return []
    allowed = {t.lower() for t in types}
    out, kept = [], 0
    for item in find_image_parts(obj):
        got = image_b64(item[2])
        if got is not None and (got[1] or "").lower() in allowed and kept < limit:
            kept += 1
            continue
        out.append(item)
    return out
```

`gateway/providers/ds4.py`, on `DS4Provider`, as class attributes next to the `wants_vision` comment:

```python
    # ds4-server's native image limits (ds4_server.c "at most 16"; server_image_media_type):
    # the rest of a native route's image parts go to the describe sidecar (vision.native_split).
    native_image_limit = 16
    native_image_types = ("image/png", "image/jpeg", "image/jpg")
```

`gateway/server.py:1637`:

```python
                img_parts = (vision.find_image_parts(obj) if prov.wants_vision
                             else vision.native_split(obj, getattr(prov, "native_image_limit", 0),
                                                      getattr(prov, "native_image_types", ())))
```

`gateway/server.py:1771-1772`: replace "Low reachability today (Ornith-1.5's native vision means img_parts is empty)" with "Reachable on a native-vision ds4 route only for the parts ds4 cannot take (vision.native_split: past 16 images, or not PNG/JPEG)".

- [ ] **Step 4: Run it to verify it passes**

Run: `python3 tests/test-suite-vision-native.py; echo rc=$?`. Expected: `6/6`, `rc=0`.

Then run `python3 tests/run-success-gate.py --tier H > <scratch>/g7.log 2>&1; tail -5 <scratch>/g7.log`. Expected: all H green, with the known worktree-only reds `H.scratchpad_tracked` and journey F81 excepted (memory: gateway-repos-layout).

- [ ] **Step 5: Commit**

```bash
git add gateway/vision.py gateway/providers/ds4.py gateway/server.py tests/test-suite-vision-native.py tests/run-success-gate.py
git commit -m "gateway: native-vision ds4 routes get their first 16 PNG/JPEG images; the rest go to the sidecar"
```

---

### Task 8: `L.vision_capability` verifies ds4 routes by their `--vision` argument

**Files:**
- Modify: `tests/test-suite-vision-capability.py:255-290` (route loop), `:360-380` (the "both kinds" case), `:590-615` (controls)

**Interfaces:**
- Produces: `ds4_vision_declared(route_row) -> bool`. It is true when `process_command` holds `--vision <path>` and the file starts with `GGUF` and its `general.architecture` is `clip`.

- [ ] **Step 1: Write the failing control**

In the controls section, add a mutation that copies an enabled ds4 route, removes `--vision` and its argument from `process_command`, and declares `vision: true`:

```python
            ds4_route = next(((m, r) for m, r, s in routes(registry) if r == "ds4"
                              and s.get("enabled") is not False), None)
            if ds4_route:
                def strip_vision(d, route=ds4_route):
                    row = d["models"][route[0]]["runtimes"]["ds4"]
                    cmd = row["process_command"]
                    if "--vision" in cmd:
                        k = cmd.index("--vision")
                        del cmd[k:k + 2]
                    row.setdefault("capabilities", {})["vision"] = True
                rc, _out = spawn("ds4_true_without_flag", strip_vision)
                case("ds4 route khai vision mà không có --vision ⇒ đỏ (rc=1)", rc == 1, "rc=%d" % rc)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 tests/test-suite-vision-capability.py; echo rc=$?`. This reads the live registry read-only, as it does today, and it is Tier L.

Expected: the new case is ❌ `rc=0`, because the checker skips ds4 routes (`:265-273`).

- [ ] **Step 3: Implement**

Add near the top:

```python
def ds4_vision_declared(row):
    """A ds4 route sees images iff its argv loads an mmproj: `--vision <file>` with a GGUF whose
    general.architecture is `clip`. This is ds4's own load contract (ds4.c --vision)."""
    cmd = row.get("process_command") or []
    if "--vision" not in cmd or cmd.index("--vision") + 1 >= len(cmd):
        return False
    path = os.path.expanduser(cmd[cmd.index("--vision") + 1])
    try:
        with open(path, "rb") as f:
            head = f.read(4 << 20)
    except OSError:
        return False
    return head[:4] == b"GGUF" and b"general.architecture" in head and b"clip" in head
```

In the route loop, replace the `runtime != "omlx"` skip for ds4:

```python
            if runtime == "ds4":
                want = ds4_vision_declared(row)
                checked += 1
                case("khai vision khớp --vision của ds4 · %s" % label,
                     isinstance(declared, bool) and declared == want or (declared is None and not want),
                     "--vision %s ⇒ vision phải là %s; đang khai %r" % ("có" if want else "không", want, declared))
                continue
```

`row` is the route's runtime dict. Use the name the loop already binds; read `:240-256` first.

The "both kinds on disk" case (`:375`) uses only oMLX types. Every served route may legitimately be vision-capable, so the union of oMLX types and ds4 argv can lack "llm". Downgrade it to informational, since the fixtures already prove both outcomes: print `ℹ` and do not `case()` it when the fixture cases passed. Ledger this as a ruling (cost if wrong: a box with only vision models loses one meaningfulness check that the fixtures already cover).

The `true_on_llm` control (`:602`) may pick a ds4 route. With the ds4 oracle it now fires: declaring `true` on a ds4 route without `--vision` is red. If no text-only route exists at all, skip it with `ℹ`, because `ds4_true_without_flag` covers it.

- [ ] **Step 4: Run it to verify it passes**

Run: `python3 tests/test-suite-vision-capability.py; echo rc=$?`

Expected:
- Before Task 10 the Ornith rows declare nothing and have no `--vision`, so the ds4 cases pass.
- The Qwen3.8 ds4 rows declare `true` with `--vision`, so they pass.
- `rc=0`, unless the oMLX-only discovery case still fails for an environment reason. If it does, record its message.

- [ ] **Step 5: Commit**

```bash
git add tests/test-suite-vision-capability.py
git commit -m "test(L.vision_capability): ds4 routes are verified by their --vision mmproj, with a planted-mistake control"
```

---

### Task 9: Quality against the sidecar (pre-registered)

**Files:**
- Create: `speed-bench/ornith-vision/QUALITY-PRE-REGISTRATION.md`, `speed-bench/ornith-vision/quality.py`, `speed-bench/ornith-vision/QUALITY.md`

- [ ] **Step 1: Pre-register before any model output is read**

Look at each image (Read tool), then write 3-5 facts per image into `QUALITY-PRE-REGISTRATION.md` as lower-case substrings with alternates (`a|b`). Commit it before Step 3. The images:
1. `fixtures/newspaper.jpg`
2. `fixtures/code.png`
3. `tests/vision-fixtures/glm53/screenshot.png`
4. `tests/vision-fixtures/glm53/diagram.png`
5. `tests/vision-fixtures/glm53/earth.jpg`
6. `fixtures/vietnamese.png`

The question for every image: "Describe this image in detail, including any text you can read."

- Score: the number of facts found in the final answer.
- Rule: native total ≥ sidecar total ⇒ PASS.
- Reported per image, and the total decides.
- The latency of each arm is reported, not gated.

```bash
git add speed-bench/ornith-vision/QUALITY-PRE-REGISTRATION.md
git commit -m "pre-registration: Ornith native vision vs the describe sidecar (facts frozen before any run)"
```

- [ ] **Step 2: Write `quality.py`**

- **Native arm:** the scratch ds4 from Task 6's launcher (with `--vision`), image as a data URI.
- **Sidecar arm:**
  1. Start a second sidecar from the repo: `MLX_VISION_PORT=8182 /opt/homebrew/bin/python3.12 ~/Documents/GitHub/AI-Gateway-MLX/scripts/vision-sidecar.py`.
  2. Get the description through `vision.describe_image` with `config.VISION_URL` pointed at `:8182`.
  3. Send Ornith (same scratch ds4) a text-only request whose user content is the gateway's exact wrapper, `[Attached image — described by the vision model (the main model cannot see pixels directly): <desc>]`, followed by the question.
- Both arms use `temperature 0` and `max_tokens 600`.
- Write `quality-<date>.json` with facts found per image and arm, TTFT, and answer length (no answer text).
- SIGTERM both processes at the end.

- [ ] **Step 3: Run it** with the slot held down and the quiet window checked. Expected: a JSON file and a printed total per arm.

- [ ] **Step 4: Write `QUALITY.md`** with the verdict under the frozen rule, a per-image table and the latencies. Commit:

```bash
git add speed-bench/ornith-vision/quality.py speed-bench/ornith-vision/QUALITY.md speed-bench/ornith-vision/quality-*.json
git commit -m "speed-bench: Ornith native vision vs sidecar quality (pre-registered facts)"
```

If the verdict is FAIL, stop before Task 10 and report to the user. The rule decides whether the route flips.

---

### Task 10: Deploy

**Files:**
- Create: `speed-bench/ornith-vision/DEPLOY.md` (ds4-metal), `reports/ornith-vision-2026-10-10/DEPLOY.md` (AI-Gateway-MLX)
- Modify: `config/runtime-registry.ornith-1.5.json` (AI-Gateway-MLX)

- [ ] **Step 1: Final reviews and merges**
  - Run superpowers:requesting-code-review on both branches. Fix the Critical and Important findings with RED→GREEN tests.
  - Merge `feature/ornith-vision` into ds4-metal `develop` and push to `origin`.
  - Merge `dongnh311/ornith-vision` into AI-Gateway-MLX `develop` and push to `origin`.
- [ ] **Step 2: ds4 production install**, per `docs/DEPLOY_AI_GATEWAY.md` rule 6:
  1. Run the quiet window.
  2. Hold `/tmp/ds4.lock`.
  3. SIGTERM the slot.
  4. Run `./deploy-ai-gateway.sh install prod/ornith-vision-<date>`.
  5. Smoke on a scratch port with `DS4_LOCK_FILE`. The rule-6 smokes, plus C1 from Task 6 against the installed binary.
  6. Release the holder.

  Expected: smokes ok, C1 PASS.
- [ ] **Step 3: mmproj and registry**
  1. Copy the mmproj to `~/.local/share/ai-gateway/ds4-models/` and check its sha256 against Global Constraints.
  2. Back up `~/.local/ai-gateway/runtime-registry.json` to `runtime-registry.json.bak-ornith-vision-<stamp>`.
  3. With one atomic write (temp file + `os.replace`), append `--vision <mmproj>` to both Ornith rows' `process_command` and set `capabilities.vision: true`.
  4. Mirror the same change in `config/runtime-registry.ornith-1.5.json` and commit it.
- [ ] **Step 4: Gateway re-stage and Tier L**
  1. Run `scripts/restage.sh`, which waits for quiet.
  2. Run `python3 tests/run-success-gate.py --tier L`.

  Expected: `L.vision_capability` green, plus the rest as before (16/17 with only the known vision step red before this change, so 17/17 is the target).
- [ ] **Step 5: Live check.** Run one real image request through the gateway with the permanent key from loopback (Task 6's C1 image). The analytics row shows no sidecar describe. That means:
  - no `vision` admission;
  - `gateway.log` shows the ds4 request with an image;
  - the reply contains the headline.
- [ ] **Step 6: Field-frequency follow-up (spec §7.4).** Write the follow-up into the gateway receipt: on or after 2026-10-17, count `request images=` lines against Ornith request lines in `gateway.log`, plus the sessions whose image count reaches 16. Report both to the user, to decide on an image-keyed disk KV cache. Counts only, never prompt text.
- [ ] **Step 7: Receipts.** Write both `DEPLOY.md` files: the build, smokes, registry diff (paths only), Tier L result, and rollback. Rollback is: restore the registry backup (the gateway reads it live, and the sidecar resumes on the next image); the binary may stay. Commit and push both repos (rule #12: own paths only).
