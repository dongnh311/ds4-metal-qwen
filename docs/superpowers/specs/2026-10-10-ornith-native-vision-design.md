# Ornith native vision — design (2026-10-10)

## 1. Why

The gateway's Ornith routes are named `…-vision-mtplx`, but the deployed model cannot see.
- **The weights.** The GGUF ds4 serves (`Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`)
  has 753 tensors, all `blk.*`, `token_embd`, `output` and `output_norm`. No vision tower ships with it. The MLX
  build that had one ran on oMLX and was deleted on 2026-09-28 (`AI-Gateway-MLX/reports/ornith-ds4-swap-2026-09-28/RESULT.md:24`).
- **The engine.** ds4 refuses `--vision` for qwen35moe: "Ornith-1.5-35B-A3B runs on single-host Metal only;
  --vision is not supported" (`ds4.c:74154-74175`). Its allowed list omits it (`ds4.c:74225-74233`).
- **The detour.** Images for Ornith go through the gateway's sidecar instead:
  - Qwen3-VL-4B describes each image in text (`gateway/server.py:1779-1822`, `gateway/vision.py:113`).
  - It costs 25-45 s per image and about 3.1 GB, and unloads after 900 s idle.
  - The 35B model reads a 4B model's paraphrase, never the pixels.

## 2. Evidence that it can work (spike, 2026-10-10, throwaway)

- **Projector.** `bartowski/Ornith-1.5-35B-A3B-GGUF` publishes `mmproj-Ornith-1.5-35B-A3B-f16.gguf`.
  - 899,283,296 bytes; sha256 `815c9a67…3c7c` was checked on download.
  - Local copy: `~/orca/workspaces/ds4-metal-data/gguf/ornith-mmproj/`.
  - It is a `clip` / `qwen3vl_merger` GGUF with 27 blocks, width 1152, 16 heads, patch 16, merge 2, no deepstack
    layers, and `projection_dim` 2048.
  - Apart from `projection_dim` (2560 there), this is the same metadata and the same 334 tensor names as the
    Qwen3.8-Flash-Next mmproj that ds4 already runs (`DS4_VISION_QWEN4`).
  - The base model's `vision_config` agrees: `deepstack_visual_indexes: []` and `out_hidden_size: 2048`.
- **Fit with the abliterated build.** llama.cpp `b1-311d421` ran it against the deployed GGUF and the mmproj, with the
  gateway's ds4 slot held down. Input: `tools/mtmd/test-1.jpeg` (a newspaper front page). Settings: `--temp 0 --reasoning off`.
  - Read correctly: the paper (The New York Times), the date (Monday, July 21, 1969), volume and number, "MEN WALK ON
    MOON", and three sub-heads. One sub-head had a one-word slip ("Eagle Has Lands").
  - Timing: prompt 486 tokens (image included) in 0.86 s; generation 60 tok/s; 9.2 s wall time including model load.
- **So the projector fits the abliterated weights.** Abliteration and calibration did not break the image-embedding
  interface.

## 3. Goal and non-goals

**Goal.** Ornith on ds4 accepts images natively, through the same image paths Qwen3.8 already uses:
- OpenAI `image_url`, Responses `input_image`, and Anthropic `image` blocks, including inside tool results.
- Both Ornith gateway routes declare `vision: true`, so the gateway stops captioning for them.

**Non-goals.**
- Video.
- Grounding / box outputs.
- An image-keyed disk KV cache (see §6.4).
- Changing the sidecar for other models.
- A new encoder: the Qwen3.8 encoder is reused as is.

## 4. Design — ds4

Six changes. Each is small because the multimodal plumbing (request parsing, image markers, encoding, spans, vision
state, the KV-cache guard) is already model-independent. What is missing is the Ornith graph's use of it.

### 4.1 Engine load
- Drop `--vision` from qwen35moe's refusal list (`ds4.c:74166`). Admit qwen35moe in the `--vision` allowed list.
- Bind with `qwen4_vision_weights_bind`. Its `projection_dim == DS4_N_EMBD` check then enforces 2048 for Ornith
  (`ds4.c:74258`).
- Set `vision_kind = DS4_VISION_QWEN4`.
- Image tokens resolve by vocabulary lookup (`<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>`;
  `ds4.c:74622-74624`). In Ornith's vocab these are 248053, 248056 and 248054, matching the base config.
- Every other qwen35moe refusal stays.

### 4.2 Prompt rendering
- `request_tokenize_multimodal_prompt` (`ds4_server.c:4405`) splits the rendered text at each image marker and appends
  `<|vision_start|>` + N × `<|image_pad|>` + `<|vision_end|>`. That is the Qwen3.5 chat template's image form.
- `render_ornith_chat_prompt_text` must carry the markers through unchanged, for user content and for tool results.
  - The plan verifies this with a test per role.
  - If it rewrites content (escaping, truncation), the renderer gets the same pass-through the Qwen3.8 renderer has.

### 4.3 Graph inputs and positions
Ornith's attention layers already rope with `pos3` (sections `[11,11,10,0]`); text rows simply write `(p, p, p)`.
Images need two things on the host side, both mirroring `qwen4_graph_stage_inputs` (`ds4.c:61732-61755`):
1. **Rows.** At an image position, `R` gets the span's embedding row instead of `token_embd[<|image_pad|>]`
   (`qwen4_span_row`).
2. **Positions.** `qwen4_mrope_pos` gives `(t, t+row, t+col)` inside an image and `p + delta` after one, with
   `g->mrope_delta` carried across chunks.

Every place that writes Ornith `pos3` must apply `delta`:
- `qwen35_graph_stage_inputs` (`ds4_qwen35moe.inc:202-213`);
- the MTP block's staging (`ds4_qwen35moe.inc:1015-1023`);
- the decode and verify paths.

Wiring:
- The qwen35 sync branch sets `g->vis_spans` / `g->vis_span_count` from `s->sync_images` around its prefill loop,
  exactly as the qwen4 branch does (`ds4.c:79493-79494`, `:79533-79534`).
- `qwen35_graph_reset` zeroes `mrope_delta`.

### 4.4 MTP
The MTP block reads `embed(token_t)` at every position (`ds4_qwen35moe.inc:1017`). For image positions it reads the
span's row, the same row the trunk saw. This mirrors how speculative decoders that support multimodal inputs build the
draft's input embeddings.
- The spec's test is **losslessness**: with MTP on, the generated tokens after an image prompt are identical to those
  with MTP off (temperature 0).
- Acceptance rate is reported, not gated. If it collapses after images, the plan records the placeholder-embedding
  alternative as a follow-up.

### 4.5 Rewind point (shipped 2026-10-09)
- **The mark saves `mrope_delta`, and the restore puts it back.** Without this, a re-send after an image would rope
  every later token at the wrong position.
- **A rewind-point hit requires the request's image spans to match the live ones**
  (`ds4_session_vision_state_matches`). A request with different images falls through to the normal paths.

### 4.6 Disk KV cache
- This needs no change, but the cost is stated. Image-conditioned state is never written to the text-keyed disk cache
  (`ds4_server.c:12222`, `:12588`). A session that holds an image therefore gets no new disk checkpoints for the rest
  of its life.
- That is today's rule for Qwen3.8 and it stays. The in-memory paths still apply: the live session and the rewind
  point.
- §7 measures how often real sessions carry images, to decide whether an image-keyed disk cache is worth a later
  round.

## 5. Design — gateway

1. **Registry.** On both Ornith ds4 rows:
   - append `--vision ~/.local/share/ai-gateway/ds4-models/mmproj-Ornith-1.5-35B-A3B-f16.gguf` to `process_command`;
   - set `capabilities.vision: true`.

   `DS4Provider.wants_vision` then goes false and the sidecar step is skipped (`gateway/providers/ds4.py:54-56`).
2. **Image budget.** ds4 rejects a request with more than 16 images (`ds4_server.c:4417`). A long Claude Code session
   can pass that. Rule, chosen to keep the prompt prefix stable:
   - The **first 16 images of a conversation go native**. Later ones go through the sidecar as text.
   - Converting an *old* image to text would change the prefix and force a full re-prefill. So the cut is by
     arrival order, never by recency.
3. **Clean-ups.**
   - The stale comment at `gateway/server.py:1771` ("Ornith-1.5's native vision means img_parts is empty") becomes true
     again; reword it to name the route flag.
   - `L.vision_capability` (`tests/test-suite-vision-capability.py`) skips ds4 routes and so cannot catch a wrong
     `vision: true` (red since 2026-09-28). It learns to read a ds4 route's `--vision` argument: `vision: true` without
     `--vision` is a failure.

## 6. Risks

| Risk | Handling |
|---|---|
| Positions off by delta somewhere (decode, MTP, rewind) | The tests in §8 compare multi-turn and MTP runs token for token; any missed `pos3` writer breaks them |
| The abliterated LM reads images worse than the base | §7 quality set; the spike already read a dense page correctly |
| Memory: +0.9 GB weights plus encoder scratch on the 512K slot | Measured at deploy; f16 is kept, because ds4's loader also takes Q8_0 if needed |
| Long sessions lose disk checkpoints after the first image | Stated in §4.6; frequency measured in §7 |
| Text-only behaviour regresses | §8.1: text-only prompts must be bit-identical before and after (`delta = 0` and `R` rows unchanged when no spans) |

## 7. Measurement

The pass criteria are frozen here.
1. **Correctness, end to end, on the production argv.** Two images: the newspaper and a code screenshot with known
   text.
   - The reply must contain the facts in a fixed list: newspaper name, date, headline; a function name and a string
     literal from the screenshot.
   - ds4 must return the same listed facts llama.cpp returns for the same inputs.
2. **Quality versus the sidecar.** A set of six images: newspaper, code screenshot, UI screenshot, chart, photo, and a
   Vietnamese-text image.
   - Each has 3-5 fixed facts written before the run.
   - The native route must recover at least as many facts in total as the sidecar route.
   - Reported per image; the total decides.
3. **Latency.** Time to first token for a one-image request, native against sidecar (sidecar loaded and unloaded
   reported separately). Reported, not gated.
4. **Field frequency.** After a week, the share of real Ornith requests that carry images, and the share of sessions
   that reach the 16-image cut. Reported for the §4.6 follow-up decision.

## 8. Testing

1. **Text-only identity.** The existing Ornith equivalence tests and the rewind-point test pass unchanged. A text prompt's
   logits are bit-identical with and without `--vision` loaded.
2. **Positions (CPU, no model).** Unit tests for Ornith staging with synthetic spans (the `DS4_QWEN4_FAKE_IMAGE` style),
   checking the rows taken and the `(t, h, w)` triples:
   - inside an image;
   - after one;
   - across a chunk boundary;
   - across two images.
3. **Multi-turn equivalence (model).** An image in turn 1 and a text question in turn 2, served as a live continuation,
   gives the same tokens as the two turns sent cold as one prompt.
4. **MTP losslessness (model).** An image prompt generates the same tokens with and without `--mtp`.
5. **Rewind point with an image (model).**
   - A re-send of an image conversation hits the rewind point and replies identically.
   - A request with a different image does not hit it.
6. **Server.** Request parsing and markers for user and tool-result images, through the Ornith renderer.
   - More than 16 images is still refused by ds4.
   - The gateway applies the 16-image cut by arrival order (a hermetic Tier H case).
7. **Gateway Tier H.** `vision: true` routes skip the sidecar. `L.vision_capability` fails on a ds4 route that declares
   vision without `--vision`.

## 9. Rollout and rollback

- **ds4.** The usual deploy (`docs/DEPLOY_AI_GATEWAY.md` rule 6):
  - quiet window, hold `/tmp/ds4.lock`, install, smoke with `DS4_LOCK_FILE` on a scratch port, release.
  - The binary change is inert until a row passes `--vision`.
- **Registry.** Copy the mmproj into `ds4-models/`, then add `--vision` and `vision: true` to both Ornith rows in one
  atomic write, after a backup. Then run Tier L.
- **Rollback.** Restore the registry backup. The gateway reads it live, and the sidecar path resumes on the next image.
  The binary can stay.
