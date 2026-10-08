# Knowledge Injection Hook — Phase 2 design

**Date:** 2026-10-08
**Parent spec:** `2026-10-08-ornith-knowledge-layer-design.md` (Path A, approved)
**Repo split:** this spec lives in `ds4-metal`; the implementation lives in `AI-Gateway-MLX`. The gateway sits in front of both inference engines, so the hook is written against the gateway request path, not against ds4 or omlx.

---

## 1. Goal

Phase 1 built the corpus, the provenance, and the measurement harness. **Nothing reads the corpus on the request path.** Phase 2 closes the last arrow in parent spec §5 — `retrieve top-K → deterministic filter → context injection → ORNITH` — and produces the numbers the pilot gate needs to decide whether to scale the corpus.

The deliverable is not "retrieval exists". It is: *an opted-in turn gets knowledge from the index, and the effect on correctness and total task latency is measured.*

---

## 2. What Phase 1 delivered (evidence, not recollection)

| Piece | Where | State |
|---|---|---|
| Per-chunk provenance columns | `gateway/context/store.py` (`corpus`, `source`, `doc_id`, `version_hash`, `language`, `generated_by`), schema v4 | live |
| Provenance on every hit | `gateway/context/retrieval.py:97` `query()` returns the provenance keys | live |
| Corpus manifest + ingestion | `gateway/context/corpora.py` | live, unused by the request path |
| Agentic trajectory chunking | `gateway/context/trajectories.py` (`schema` / `attempt` / `correction` / `outcome`) | live |
| Pre-index verification | `gateway/context/verify.py`, wired into `indexer.index_project(..., verify_fn=…)` | live |
| Receipt shape + validation | `evals/knowledge_baseline.py` | live |
| Live speed baseline | `evals/live_measure.py`, artifact `evals/artifacts/ornith-prefill-m5pro-ornith-ice-23g-ds4.json` | live: 95.6–96.3 tok/s decode, 1249–1353 tok/s true prefill, 337–365 ms TTFT, 4 runs |
| Cap sweep, ablation, useful-retrieval proxy, pilot gate | `evals/cap_sweep.py`, `evals/ablation.py`, `evals/retrieval_quality.py`, `evals/pilot_gate.py` | live, **consume `correct` values nothing produces yet** |
| Index size | 15,116 chunks / 196 projects | live |

The gap is exactly one thing: the seam.

---

## 3. A deviation from parent spec §5, ruled explicitly

Parent §5 says *"Three separate indices rather than one, because retrieval policy differs by domain."* Phase 1 built **one** index with a `corpus` column.

**Ruling:** keep one physical index; select the domain by filtering the `corpus` column and applying per-domain policy at query time (k, path boost, diversity cap). Three indices would triple the schema, the rebuild cost, and the number of places a provenance bug can hide, for a benefit that a `WHERE corpus = ?` clause delivers.

**Cost if wrong:** if a domain ever needs a genuinely different tokenizer or index structure (Vietnamese morphology is the candidate), the split has to happen later and the index is disposable anyway (`store.py:47-68`).

---

## 4. Decisions locked by the partner

These are not open. Every task in the plan must respect them.

1. **Gate:** OFF by default. Opt-in per request via header, plus an env default. No existing traffic changes behaviour.
2. **Branch:** BM25-only. The hook calls `retrieval.query(..., no_embed=True)`. No embedder runs on the hot path — it competes with decode for the same 307 GB/s.
3. **Carrier:** a separate block with its own formatter and its own idempotent strip, position configurable, default `fold`.
4. **Pilot corpus:** the 196 already-indexed projects plus analytics-derived trajectories. Vietnamese corpus is Phase 3, not a gap in this plan.

---

## 5. The seam

The memory prefetch block at `gateway/server.py:1652-1697` is the template: soft-fail, before the ctx check so injected tokens are counted and compacted if they push over, `changed` and `est_val` updated, a `tracking.bump` for the denominator.

The hook mirrors that shape and lives immediately after it:

```
_knowledge_enabled, _corpora = context.inject.policy(self.headers, obj)
if _knowledge_enabled:
    _knowledge_ms = None
    _knowledge_injected = 0
    try:
        _q = memory.recall_query(obj, config.KNOWLEDGE_ANCHOR)
        if len(_q) >= config.KNOWLEDGE_MIN_QUERY_CHARS:
            _t0 = time.perf_counter()
            _hits = context.inject.retrieve(_q, _corpora, config.KNOWLEDGE_CAP)
            _knowledge_ms = (time.perf_counter() - _t0) * 1000
            if context.inject.block(obj, _hits, position=config.KNOWLEDGE_POSITION):
                changed = True
                _est_before = est_val
                est_val = _est_or_fail_closed(obj)
                _knowledge_tokens = est_val - _est_before
                _knowledge_injected = 1
                tracking.bump("knowledge_inject")
    except Exception:
        pass
    # _knowledge_ms / _knowledge_injected are already initialised above, so a
    # swallowed exception still records "ran, injected nothing" rather than nothing at all.
```

The locals are initialised alongside `_memory_injected = None` at `server.py:1312` and passed to `track_end(...)` at `server.py:2119`, exactly as `memory_injected` is today (`tracking.py:194, 235-236` → `analytics.record_event`, `analytics.py:449, 490`). The hook never calls `record_event` itself — the finish path does.

Three ordering constraints, each pinned by a test:

- **Before the ctx check.** Injected tokens must be counted, or a cap of 4000 can push a request over the context guard and the guard will not see it.
- **After memory prefetch.** Two blocks, each with its own strip. Order is fixed so a test can assert both are present and neither stacks.
- **Soft.** Any retrieval failure leaves the request byte-identical. A corpus miss is never a 500.

`memory.recall_query` is reused, not reimplemented — it already decides which turn the recall runs against.

---

## 6. Config surface

| Env | Default | Why that default |
|---|---|---|
| `MLX_KNOWLEDGE_INJECT` | `0` | Off until the pilot gate says otherwise. A hot-path behaviour change with no denominator is how the memory-prefetch regression happened. |
| `MLX_KNOWLEDGE_CAP` | `0` | Parent §10: no cap is hard-coded. The sweep chooses it. A default cap before the sweep is a guess shipped as a constant. |
| `MLX_KNOWLEDGE_POSITION` | `fold` | `head` re-prefills the whole context (measured 51.5% hit, 24.7→72.1% swing). `fold` keeps every earlier message byte-identical *and* recalls per turn. |
| `MLX_KNOWLEDGE_CORPORA` | `code` | The only corpus that exists. |
| `MLX_KNOWLEDGE_MIN_QUERY_CHARS` | `24` | A three-token query retrieves noise. The value is an open measurement, not a settled one — see §13. |
| `MLX_KNOWLEDGE_ANCHOR` | `latest` | Same as memory prefetch. `first` is available for byte-stability experiments. |

Precedence, stated so there is no second reading of it:

| env | header | result |
|---|---|---|
| `0` | absent | off |
| `0` | `X-Knowledge-Corpus: code` | **on**, corpora = `code` |
| `1` | absent | on, corpora = `MLX_KNOWLEDGE_CORPORA` |
| `1` | `X-Knowledge-Corpus: none` | **off** — explicit opt-out beats the global default |

The header is the per-request lever; env is the global default. That is what lets the sweep run all six cap arms against one running gateway without restarting it: the harness varies the header, not the process.

`X-Knowledge-Cap: 2000` sets the cap for that request and overrides `MLX_KNOWLEDGE_CAP`. A non-numeric value is ignored and the env default applies — a malformed header must not crash the request path. `policy()` therefore returns `(enabled, corpora, cap)`.

---

## 7. Carrier semantics

`format_knowledge_block(hits)` produces:

```
[RETRIEVED KNOWLEDGE — from the local index, not verified fact]
- [code · ds4-metal · gateway/context/store.py · 3f9a1c] <chunk text>
- [agentic · analytics · tool_call:graph_callers · 7b2e04] <chunk text>
```

Rules:

- **The cap is in tokens, not chunks.** `retrieve(q, corpora, cap)` takes hits from `retrieval.query` in rank order and stops when the running estimate of the block exceeds `cap`. A cap of 500 must produce roughly 500 tokens of block, not 500 chunks. The estimate uses the same estimator the admission gate uses, so the number the sweep reports is the number the scheduler sees.
- **Bounded** like `MEMORY_LINE_MAX` / `MEMORY_BLOCK_MAX`. The block rides into later turns; an unbounded block is a permanent per-request context tax.
- **Provenance is in the text**, not only in the DB. The model cannot tell a verified chunk from a teacher-generated one unless the label is in the block. This is invariant #3 of the parent spec, and it is unenforceable if provenance stays a column the model never sees.
- **Idempotent strip.** `strip_knowledge_block()` removes an existing block wherever it sits, so re-injection replaces rather than stacks.
- **Position behaviour is asserted, not assumed.** `head` → message[0] changes every turn. `tail` → earlier messages byte-identical, standalone trailing system message. `fold` → earlier messages byte-identical, block prepended inside the newest user turn. For a tool-result or multimodal turn whose newest content is a list, fold targets the first text part; a pure tool_result turn with no text part is left alone.

---

## 8. Measurement

Add three columns to `analytics.events`, following the existing idempotent-ALTER pattern (`analytics.py:101-134`) and the precedent of the per-request memory columns already in `record_event` (`analytics.py:448-450`: `memory_prefetch`, `memory_query_empty`, `memory_injected`, `memory_block_hash`):

- `knowledge_ms REAL` — retrieval time for this request.
- `knowledge_injected INTEGER` — 1 if a block was added, 0 if the hook ran and added nothing, NULL if the hook did not run.
- `knowledge_tokens INTEGER` — the estimator delta: `est_after − est_before` from `_est_or_fail_closed` (`server.py:359`). It is an **estimate**, and the column name and the spec both say so; it is not a tokenizer count.

NULL on rows written before the change, exactly as `ended_by` and `is_canary` do. The three-way distinction in `knowledge_injected` (ran-and-empty / ran-and-injected / did-not-run) is what makes the cap=0 arm interpretable.

**This reverses a Phase 1 ruling deliberately.** Phase 1's `derive_receipt` took `retrieval_ms` as a harness-supplied argument because analytics had no such column — reading a missing value as `0.0` reported a measurement that never happened. Once the gateway writes the column, reading it is legitimate. The rule is unchanged: **never report a number the system did not measure.**

`validate_receipt` must treat `knowledge_ms` NULL as "hook off for this arm", not as "retrieval took 0 ms". The cap=0 arm legitimately has no retrieval.

---

## 9. Eval suites

Two suites, both with hand-authored gold answers stored as data, not generated by a model:

**9.1 Code QA.** Questions drawn from the user's own repositories — where is X handled, what is the concurrency limit, which module owns Y. Gold answer is a short factual string plus the file that proves it. **30 cases**, sampled across at least 6 of the 196 indexed projects so the sweep is not measuring one repo.

**9.2 Agentic tool use.** Multi-step tasks against the real MCP schemas, built from `context.trajectories` chunks. Gold is the correct tool and argument set.

Both suites emit per-case `{question, gold, correct}` so `cap_sweep.sweep()` and `pilot_gate.decide()` receive the `correct` rate they already expect. No new interface.

**Deferred:** the Vietnamese QA suite. It is Phase 3, gated on the pilot gate's verdict. Shipping Phase 2 without it is not a gap in Phase 2; it is the phasing the parent spec §15 already states.

---

## 10. The sweep and the gate

Run against the live gateway, ds4 + `ornith-ice-23g`, same checkpoint, same quantization, same sampling, per `docs/PERFORMANCE.md`:

| cap | correct | prefill_tps | decode_tps | ttft_ms | knowledge_ms | total_ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | | | | — | |
| 500 | | | | | | |
| 1000 | | | | | | |
| 2000 | | | | | | |
| 3000 | | | | | | |
| 4000 | | | | | | |

The chosen cap is the **smallest** that meets the quality target inside the latency budget. Decode tok/s is deliberately not the deciding metric: it is already ~96 tok/s, and a cap that raises decode tok/s while doubling total task latency is a regression.

The gate outputs one of three verdicts: scale, hold, or abandon. It refuses to scale on either no uplift or a latency blowout — that asymmetry is already implemented in `evals/pilot_gate.py`.

---

## 11. Non-goals

- No weight change, no training, no architecture change. Ornith's GGUF is immutable.
- No LLM call in the production retrieval path. The judge is eval-only.
- No embedder on the hot path.
- No second retriever. `gateway/context/retrieval.py` is the only query path.
- No corpus content generation in this phase.

---

## 12. Review focus — failure modes no happy-path test covers

1. **cap = 0 is a true no-op.** The request must be byte-identical, not carry an empty block that perturbs the prompt and defeats the prefix cache.
2. **Precedence in both directions.** env off + header present → on; env on + header `none` → off. Both rows of the §6 table are pinned by a test, not by convention.
3. **Injection position changes prefill cost.** Total latency can worsen while decode tok/s stays flat. The receipt must show both.
4. **Retrieval raises.** The request must be untouched and the failure counted, not silently swallowed into "no relevant knowledge exists".
5. **Two blocks interact.** Memory block and knowledge block must not stack, and stripping one must not strip the other.
6. **Injected tokens counted before the ctx check.** If they are added after, a large cap can push a request over the guard invisibly.
7. **Empty or punctuation-only query.** `retrieval.query` must return `[]`, not raise.
8. **`knowledge_ms` NULL vs 0.** A hook that is off is not a hook that retrieved in zero milliseconds.
9. **Module allowlist.** New import edges (`server -> context.inject`, `context.inject -> context.retrieval`) must be registered in `docs/MODULE-ALLOWLIST.md` or rule #16's gate goes red.

---

## 13. Open measurements

- `KNOWLEDGE_MIN_QUERY_CHARS = 24` is a guess. Measure the useful-retrieval rate against query length before trusting it.
- Whether `fold` really preserves the prefix cache on ds4 (the 51.5% figure was measured on mlx_lm) — the sweep's prefill column answers this.
- Whether the diversity cap (`cap = max(2, k // 3)` in `retrieval.py`) is right for a mixed corpus, or wastes slots on one project.

---

## 14. Success criteria

- An opted-in request demonstrably contains corpus chunks with provenance in the prompt.
- The cap sweep produces a receipt that validates, with a real `knowledge_ms` per arm.
- The pilot gate returns a verdict from live data, not from a fixture.
- Tier-H gate green, with the two pre-existing failures unchanged.
- Decode throughput within 5% of the 95.6–96.3 tok/s baseline.
- Nothing on the request path changes for a client that does not send the header.

---

## 15. What this spec does not claim

- It does not claim the knowledge layer improves answers. That is what the sweep decides; the code only makes the measurement possible.
- It does not claim Vietnamese coverage. That corpus does not exist yet.
- It does not claim the cap is known. `KNOWLEDGE_CAP` defaults to 0 precisely because it is not.
- It does not claim the hook is engine-independent in the sense of being written for both engines — it is written for the gateway, which is the layer that is engine-independent by construction.
