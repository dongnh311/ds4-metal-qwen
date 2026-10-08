# Ornith Knowledge Layer — Design Spec

**Date:** 2026-10-08
**Path:** A — retrieval layer over an existing backbone, no weight change
**Status:** Architecture approved; implementation plan not yet written

---

## 1. Goal

Raise answer quality of **Ornith-1.5-35B-A3B** on three domains — code, Vietnamese, agentic tool use — by attaching a knowledge retrieval layer.

Hard constraints:

- No weight change to the backbone.
- No training.
- No architecture grafting.
- No embedder in the retrieval path.

Decode speed is already solved. The open problem is knowledge coverage, and the metric that matters is end-to-end task latency and answer correctness, not tok/s.

---

## 2. Source of truth for the deployed checkpoint

**GGUF metadata of the deployed checkpoint is authoritative.** Architecture is never inferred from a model card, a repo README, or a similarly named upstream model.

Verified from `Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`:

| Property | Value | Metadata key |
|---|---:|---|
| Architecture | `qwen35moe` | `general.architecture` |
| Layers | **41** | `qwen35moe.block_count` |
| Experts | 256 total, 8 active | `expert_count`, `expert_used_count` |
| Hidden size | 2048 | `embedding_length` |
| Expert FFN | 512 | `expert_feed_forward_length` |
| Shared expert FFN | 512 | `expert_shared_feed_forward_length` |
| Attention | hybrid: full attention every 4th layer, linear/SSM otherwise | `full_attention_interval`, `ssm.*` |
| MTP | 1 nextn layer | `nextn_predict_layers` |
| Native context | 262,144 | `context_length` |
| File size | 22.84 GB | on-disk |

This matters because steering shapes, LoRA targets, and runtime layer indexing are all derived from these numbers. An upstream model card describing a 40-layer variant is not the checkpoint that runs here.

---

## 3. Architectural decision: serving engine ≠ evaluation engine

Two runtimes execute Ornith on this machine, and they are **not** interchangeable:

| Role | Engine | Measured decode | Evidence |
|---|---|---:|---|
| **SERVING_ENGINE** — receives production traffic | omlx | 99.0–99.9 tok/s | `.omlx/logs/nohup-serve-ornithprod.log` |
| **EVAL_ENGINE** — runs benchmarks and eval suites | ds4-metal | 73.0–84.8 t/s | `ds4-metal-data/ornith-r2-deploy/*.log` |

Design consequence:

```
KNOWLEDGE LAYER  = engine-independent
                   (corpus format, index format, retrieval API)

INTEGRATION HOOK = engine-specific
                   (chosen in Phase 2, blocking)
```

The corpus, index, and retrieval logic must not assume either runtime. The injection hook is written against whichever engine serves traffic, and that choice is a **blocking decision before Phase 2**, not an implementation detail.

**99 tok/s is not the KPI of a knowledge-enabled system.** Once injection adds context, the measured quantities are:

- TTFT (time to first token)
- prefill tok/s
- retrieval latency (index query + rerank + I/O)
- decode tok/s
- **total task latency**

Decode throughput can stay flat while total task latency gets worse. That is the failure mode this design must detect.

---

## 4. Explicit non-goals, with evidence

| Ruled out | Evidence |
|---|---|
| Cross-architecture weight merge | Qwen3.8 = `qwen4exp` 49L/2560/512e; GLM = `kda_*`/`hc_*` 288e; DS4.1 = Engram+compressor+indexer. No shared tensor layout → no SLERP/TIES/DARE. |
| Engram/PLE graft into Ornith | `ds4_engram.c` hardcodes `DS4_ENGRAM_LAYERS/NGRAM/COLS`. PLE appears only in `qwen4exp` shape tables (`n_ple_layer`, `n_ple_ngram`, `n_ple_head_dim`). `qwen35moe` has no `ple` keys. Injection points must be trained. |
| Steering-vector transfer | Steering files are per-architecture: 48×2560 (Qwen3.8), 43×4096 (DS4), 45×4096 (GLM). Ornith is 41×2048. Directions are not transferable. |
| MiMo 2.6 Flash contribution | `grep -i mimo` over `ds4.c`, `ds4_agent.c`, `download_model.sh`, `docs/MODELS.md` = 0 hits. No kernel exists. |
| Embedder-based retrieval | Competes for the same 307 GB/s budget as decode. BM25 avoids this. |
| Distillation | Requires training. Out of scope by decision. |

**Framing carried through the whole spec:** the three teacher models are **offline data generators**, not sources of transferred capability.

```
teacher output  ≠  teacher capability
teacher output  =  textual evidence, examples, trajectories
```

A retrieved chunk from GLM is a string. It is not GLM's reasoning.

---

## 5. Architecture

```
                 ORNITH-1.5-35B-A3B
                       │
                       │  decode unchanged
                       ▲
                       │
              context injection  ← engine-specific hook
                       ▲
                       │
            retrieve top-K  →  deterministic filter
                       ▲
                       │
        ┌──────────────┴──────────────┐
        │        KNOWLEDGE LAYER      │
        │      (engine-independent)   │
        │                             │
        │  CODE      BM25 index       │
        │  VI        BM25 index       │
        │  AGENTIC   BM25 index       │
        └─────────────────────────────┘
                       ▲
                       │
              corpus builder (offline)
        Qwen3.8 / GLM5.3 / DS4.1 as generators
```

Three separate indices rather than one, because retrieval policy differs by domain: code wants exact identifier matches, Vietnamese wants terminology, agentic wants schema lookup. One index forces one policy.

---

## 6. Retrieval pipeline

### Production path (deterministic, no LLM)

```
query
  ↓
domain classification
  ↓
BM25 top-K
  ↓
score threshold
  ↓
dedup (exact + near-exact)
  ↓
entity-overlap filter
  ↓
inject
```

The production path never calls a model for reranking.

### Evaluation path (judge available, off by default)

```
query
  ↓
BM25 candidate set
  ↓
optional model judge
  ↓
measure whether reranking actually helps
```

The judge exists to answer one question: does reranking beat the deterministic filter? If it does not, it is removed. It is not allowed to become a dependency of the production pipeline.

---

## 7. Provenance schema

Every chunk carries full provenance. Without it, a benchmark improvement cannot be attributed, and stale indexes cannot be detected.

```json
{
  "corpus": "code",
  "source": "repo:foo",
  "document_id": "src/auth.py",
  "commit": "abc123",
  "chunk_id": "17",
  "version_hash": "sha256:...",
  "language": "python",
  "retrieval_score": 12.84,
  "generated_by": "qwen3.8-flash-next",
  "generated_at": "2026-10-08T00:00:00Z"
}
```

Provenance is used for three things:

1. Attributing an answer to a specific chunk during evaluation.
2. Detecting stale code or documentation.
3. Measuring which corpus contributes to which domain's uplift.

---

## 8. Corpus builder

### Code

The user's own repositories plus public code in the target languages and frameworks. Chunks aligned to structural units (function, class, module), not fixed token counts.

### Vietnamese

Wikipedia vi dump, legal and technical documents, curated domain corpora.

### Agentic

**Trajectories are kept in original form**, not flattened into question-answer pairs. A useful agentic chunk preserves:

```
tool schema → tool call → observation → correction → final state
```

The correction step is where the value is. Flattening to Q&A destroys it.

Also included: the actual MCP tool schemas used in production.

### Verification before indexing

A chunk must pass:

1. Deduplication.
2. Contradiction check against other chunks in the same domain.
3. For code: does it actually run, or is it plausibly-shaped hallucination?

Failing chunks are dropped, not down-weighted.

---

## 9. Pilot gate before scaling the corpus

Do not run three teachers over a large workload before retrieval is proven.

```
real workload query set
        ↓
small pilot corpus
        ↓
BM25
        ↓
Ornith baseline vs Ornith + RAG
        ↓
measure useful-retrieval rate + task latency
        ↓
scale ONLY if there is measured uplift
```

If retrieval does not improve the objective score on the pilot, there is no justification for spending compute or SSD on a large corpus. This gate is mandatory, not optional.

---

## 10. Retrieval cap is an experimental variable

No cap is hard-coded. The cap is chosen by a sweep:

| Retrieved tokens | Quality | Prefill cost | TTFT | Retrieval latency | Total task latency |
|---:|---:|---:|---:|---:|---:|
| 0 (baseline) | — | — | — | — | — |
| 500 | | | | | |
| 1000 | | | | | |
| 2000 | | | | | |
| 3000 | | | | | |
| 4000 | | | | | |

The chosen cap is the **smallest** value that reaches the quality target. Past some point, injected context dilutes the prompt and costs prefill for nothing.

Short turns may skip retrieval entirely; the domain classifier and a turn-length threshold decide.

---

## 11. Metrics

**1. Useful-retrieval rate.** Retrieving a chunk is not success. Two tiers:

- *Cheap proxy (all cases):* does the chunk contain a span overlapping a key entity of the gold answer?
- *Precise ablation (sampled subset):* remove one chunk at a time. If answer correctness changes, the chunk was useful. This is ground truth; it is expensive, so it runs on a sample.

A retrieved chunk that is never used counts as a **retrieval failure**, not a neutral event.

**2. Answer quality delta.** Correctness per domain, Ornith-with-layer vs Ornith-alone. Same checkpoint, same quantization, same context, same sampling.

**3. Prefill cost.** Seconds added by injected context.

**4. TTFT.** Time to first token, with and without retrieval.

**5. Total task latency.** The number the user actually experiences.

**6. Decode throughput regression.** Must stay within 5% of baseline.

---

## 12. Evaluation protocol

Follow `docs/PERFORMANCE.md`: same checkpoint, quantization, context, and sampling; record the commit; note whether weights were resident or streamed; keep other GPU workloads idle; repeat in alternating order. One favorable run is not a result.

Three suites:

1. **Vietnamese QA** — factual and domain questions with prepared gold answers.
2. **Code** — tasks drawn from the user's actual repositories, not synthetic benchmarks.
3. **Agentic tool use** — multi-step tasks against the real MCP schemas.

Baseline is always Ornith with 0 retrieved tokens.

Report retrieval metrics separately from answer metrics, so a failure is attributable: a bad answer caused by a bad index is not a model problem.

---

## 13. Risks

| Risk | Mitigation |
|---|---|
| Teacher outputs are unverified claims | Verification stage before indexing; provenance on every chunk |
| Retrieval hurts rather than helps | Ablation tier detects this; pilot gate stops scaling |
| Prefill dominates for short answers | Sweep finds the cap; short turns skip retrieval |
| Corpus staleness | Provenance timestamps; refresh cadence defined in implementation |
| Domain misclassification routes to the wrong index | Evaluate classification accuracy separately |
| Judge silently becomes a production dependency | Judge off by default; production path is deterministic |
| Index quality is the real ceiling | Report retrieval and answer metrics separately |

---

## 14. Open measurements required before implementation decisions

1. **Prefill tok/s of Ornith on the M5 Pro.** Not measured anywhere. Every retrieval-cap decision is a guess until this exists.
2. **Baseline Ornith quality** on the three suites. Without it, "improvement" is undefined.
3. **Serving engine confirmation.** Logs indicate omlx is the production path. Confirm before writing the integration hook.
4. **Actual bytes/token of Ornith.** The 3.62 GB/token figure is an upper bound derived from measured throughput, not a measurement.

Hardware reference: Apple M5 Pro (20-core GPU, 64 GB) = 307 GB/s, per [Apple MacBook Pro specifications](https://www.apple.com/macbook-pro/specs/).

---

## 15. Phasing

**Phase 0 — Baseline.** Measure prefill, decode, TTFT, and answer quality with 0 retrieved tokens. Nothing else can be decided before this.

**Phase 1 — Pilot corpus + index.** Small corpus, three BM25 indices. No model changes.

**Phase 2 — Integration hook.** Blocking decision: which engine serves traffic. Write the hook for that engine only.

**Phase 3 — Sweep.** Run the 0/500/1000/2000/3000/4000 grid. Record quality, prefill, TTFT, retrieval latency, total task latency.

**Phase 4 — Gate.** Scale the corpus only if Phase 3 shows uplift. Pick the cap. Re-run the speed regression.

---

## 16. Success criteria

- Answer quality improves on at least two of the three domains.
- Useful-retrieval rate is measured, not assumed.
- Decode throughput within 5% of baseline.
- Total task latency improves, not just tok/s.
- No weight change, no training, no architecture change.
- Production retrieval path contains no LLM call.

---

## 17. What this spec does not claim

It does not produce a new model. It produces a backbone plus a sidecar. If the requirement is literally "one GGUF file containing knowledge from four models," that requirement is not satisfiable under these constraints, and the honest answer is that it requires training.
