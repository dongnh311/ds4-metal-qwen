# Trajectory Corpus — Phase 3-A design

Parent: `2026-10-08-ornith-knowledge-layer-design.md`. Predecessor:
`2026-10-08-knowledge-injection-hook-design.md` (Phase 2, shipped 2026-10-08).

The parent spec's §15 phasing is complete: Phase 3 (sweep) and Phase 4 (gate) ran on
2026-10-08 and returned `scale: True` with a 0.750 uplift. What the gate authorized is
the next thing: **scale the corpus**. This spec covers the first of three corpus-scaling
sub-projects — the agentic trajectory corpus. The Vietnamese corpus is a separate spec
(3-B) and per-domain retrieval policy is a third (3-C).

---

## 1. The problem, stated as evidence

`gateway/context/trajectories.py` chunks an agentic trajectory into
`schema / attempt / correction / outcome` and is tested. **It has no caller.** Grepping
the whole repo for `chunk_trajectory` finds exactly two places: the module itself and
its test suite. Nothing produces events, nothing consumes chunks, nothing stores them.

So the agentic corpus is a library with no producer and no sink. Parent §"Agentic" says
the value is in the correction step — `tool schema → tool call → observation →
correction → final state` — and that flattening it to Q&A destroys it. That value is
currently unreachable by the retrieval layer, because it never exists in the index.

What the gateway does have:

| Fact | Evidence |
|---|---|
| Real agentic traffic | 31,993 of 84,325 analytics events have `finish_reason = 'tool_calls'` |
| Tool results reach the request path | `guards.truncate_tool_results` runs on them; `readpath_oversized_tool_results` is measured per request |
| No trajectory is persisted anywhere | analytics `events` has 33 columns, none of them a tool call or observation; no conversation table exists |
| The corpus filter already works | `retrieval.query(..., corpus=[...])` builds `AND chunks.corpus IN (...)` |
| The sink already works | `store.replace_file(project, path, sha, mtime, chunks, provenance)` is idempotent per `(project, path)` and stamps `corpus/source/doc_id/version_hash` |

The gap is one arrow: **request path → trajectory chunks → index**.

## 2. What this round is not

- Not the Vietnamese corpus. Different data problem, different spec.
- Not a training round. No weights change; the parent spec's hard constraints hold.
- Not a claim that trajectories help. The measurement in §6 decides that. If reachability
  is low, the corpus stays off and the finding is recorded as a negative result.
- Not a general conversation store. Capture is scoped to tool calls and their
  observations, bounded, and prunable.

## 3. Design

Four units, each with one job and a testable interface.

### 3.1 Adapter — `trajectories.events_from_messages(messages)`

Pure function. Canonical OpenAI-shaped messages in, the event list
`chunk_trajectory` already expects out.

```
[{"role":"assistant","tool_calls":[{"id":"c1","name":"graph_callers","arguments":"{...}"}]},
 {"role":"tool","tool_call_id":"c1","content":"no such project"}]
→ [{"type":"tool_call","tool":"graph_callers","args":{...},"call_id":"c1"},
   {"type":"observation","tool":"graph_callers","output":"no such project","call_id":"c1"}]
```

Rules:
- A `tool` message is paired with the `tool_call` whose `id` matches, not with the
  message that happens to precede it. Real clients interleave; positional pairing
  mislabels observations.
- Tool schemas ride in the request as `tools` (or `functions`); the adapter emits
  `tool_schema` events from them so the schema chunk exists.
- Anything unparseable is skipped, never raised. The adapter is on the request path.
- `_failed` in the existing chunker stays as-is: `error` field is authoritative, string
  markers are the fallback.

Why a separate function rather than changing `chunk_trajectory`: the chunker's contract
is already pinned by tests and is source-agnostic. Widening it to accept raw messages
would make every existing test depend on request shape.

### 3.2 Capture — `trajectories.capture(obj, conv_id)`

Returns `{"ran", "chunks", "ms", "skipped_reason"}`. Three-state, same semantics as the
knowledge hook: a capture that never ran reports NULL, a capture that ran and stored
nothing reports 0.

Sequence:
1. Bail out (before touching the DB) if `config.KNOWLEDGE_TRAJECTORIES` is 0, if
   `conv_id` is missing, or if the request contains no tool messages.
2. Build events, chunk them, bound each chunk's text at `KNOWLEDGE_TRAJ_MAX_CHARS`.
3. Hash the whole chunk set. If `files` already has this `(project, path)` with the same
   sha, return `{"ran": True, "chunks": 0, "skipped_reason": "unchanged"}` — every
   agentic turn re-sends its history, so without this the same trajectory is re-written
   on every turn.
4. Write via `store.replace_file("trajectory", conv_id, sha, now, rows, provenance)`.

**Off-thread.** The write goes to a bounded queue (200 entries) drained by a daemon
thread, the same shape `analytics` uses for its DB writes. Extraction is synchronous (it
is a pass over a list already in memory, measured in §6); the SQLite write is not. A full
queue drops the **oldest pending** entry — the recent trajectories are the ones worth
keeping — rather than blocking the request.

One identity caveat, stated rather than hidden: `conv_id` is a truncated hash of the
first user turn, so two sessions that open with the same prompt share an id and their
trajectories merge under one `doc_id`. The analytics layer already treats `conv_id` as
a conversation identity with this exact weakness; the corpus inherits it.

**Soft.** Every exception is swallowed, counted (`trajectory_capture_failed`), and
reported as a measured `trajectory_ms`. A capture crash never changes the request bytes.

Gate: capture runs independently of injection. `MLX_KNOWLEDGE_INJECT=0` means the block
is not spliced; it does not mean the corpus stops growing. Conflating the two would make
the corpus depend on a client header.

### 3.3 Bounds and retention

| Knob | Default | Why |
|---|---|---|
| `MLX_KNOWLEDGE_TRAJECTORIES` | `1` | Capture is the corpus-building step; without it the corpus is empty and 3-B/3-C have nothing to measure |
| `MLX_KNOWLEDGE_TRAJ_MAX_CHARS` | `600` | Observation excerpt. Chosen so a correction survives the cut while a 40 KB `cat` of a file does not flood the index |
| `MLX_KNOWLEDGE_TRAJ_MAX_CHUNKS_PER_CONV` | `24` | A long happy run generates phantom chunks; the cap keeps one conversation from dominating the corpus |
| `MLX_KNOWLEDGE_TRAJ_TOTAL_CHUNKS` | `20000` | Hard ceiling. When exceeded, the oldest conversation's chunks are dropped whole |

Pruning is by conversation, not by chunk, so a trajectory is either present or absent —
half a correction is worse than none.

### 3.4 Retrieval policy for `agentic`

Deferred from Phase 2 §"what this does not answer": whether `max(2, k // 3)` diversity is
right for a mixed corpus. Checked against the code: the cap is keyed on `path`
(`retrieval.py:238`), and trajectory chunks use `path = conv_id`, so the cap already
spreads conversations without any change. No new cap is needed — inventing one would be
a second mechanism doing what the first already does.

The one agentic-specific rule is a **tool hint**: a chunk about a tool the request is
holding ranks above an equal-scoring chunk that is not. The first draft keyed this on
query terms; that is redundant, because BM25 already sees a tool name in the query. The
signal retrieval genuinely cannot see is the tool catalog the client offered — a user
asking "why did that call fail" names no tool, but the request does.

A request about `graph_callers` that surfaces five chunks from five conversations about
five tools is five wasted slots, and BM25 cannot tell them apart because the tool name
is one word in a 600-character excerpt.

This is deliberately small. A full per-domain policy engine is 3-C.

### 3.5 Observability

- Counters on `/status`: `trajectory_capture`, `trajectory_chunks`, `trajectory_capture_failed`, `trajectory_queue_dropped`.
- Analytics columns, three-state like the knowledge columns: `trajectory_ms` (REAL), `trajectory_chunks` (INTEGER). NULL = capture never ran.
- `/status` reports the corpus census: chunks per corpus, so an operator can see whether the agentic corpus is growing without opening SQLite.

## 4. Data flow

```
client request (tools + history)
   │
   ├─ canonical IR (existing)
   ├─ _trajectory_capture(obj, conv_id)      ← new, soft, off-thread write
   ├─ _knowledge_push(obj, headers)          ← existing, unchanged
   └─ model
                                   │
                          store.replace_file("trajectory", conv_id, …)
                                   │
                          chunks(corpus='agentic', source='trajectory',
                                 doc_id=conv_id, version_hash=chunk sha)
                                   │
                          retrieval.query(corpus=['agentic'])
```

## 5. Failure modes

| Failure | Behaviour |
|---|---|
| Tool message with no matching call id | Skipped; counted in `skipped_reason` |
| Observation larger than `MAX_CHARS` | Excerpted, not dropped — the head of an error message is the correction |
| SQLite write fails (locked, disk) | Counter bumps, request unaffected, chunk retried on the next turn (the sha differs then, so it is not lost forever) |
| Queue full | Newest dropped, counter bumped. Capture is best-effort by design |
| Same trajectory seen 30 times | sha check makes 29 of them no-ops |
| Corpus grows past the ceiling | Oldest conversations pruned whole |
| A trajectory contains a secret | It is a local excerpt of data the client already sent. It is never committed, never sent upstream, and is bounded. The repo's no-secrets rule applies to the repo, not to the local index |

## 6. Measurement

Two steps, in order, each with a threshold that decides whether the next step runs.

**6.1 Replay harness — `evals/trajectory_replay.py`.** Runs real agentic loops through
the live gateway against the gateway's own MCP tools (`gateway/mcp.py`), with a
deterministic tool backend so a failure is reproducible. Each run produces a genuine
trajectory through the genuine capture path. Without this the corpus is empty and there
is nothing to measure.

**6.2 Offline reachability — `evals/trajectory_reachability.py`.** Leave-one-out: for
each trajectory that contains a correction, remove that correction chunk from the
index copy, query with the *failing observation*, and ask whether the correction
appears in the rendered block. Threshold: **≥ 70% of held-out corrections reachable**.
Below that, the sweep does not run and the result is recorded as a negative finding.

**6.3 Live sweep — reuse `evals/cap_sweep.py`.** Arms: cap 0, cap 2000 with
`corpora=code`, cap 2000 with `corpora=agentic`, cap 2000 with `corpora=code,agentic`.
The comparison that matters is not code-vs-nothing (Phase 2 already settled that); it is
whether agentic chunks add anything on top of code chunks. Pre-registered before arm 1,
same VOID discipline.

**6.4 What is reported separately.** Capture cost (ms/request), corpus size, reachability,
answer correctness. Never merged into one number.

## 7. Testing

Unit (no model, no live gateway):
- adapter pairs by `tool_call_id`, not position; malformed input skipped, never raised
- a happy run produces no `correction` chunks (existing rule, re-pinned through the adapter)
- excerpt bound: a 40 KB observation yields a chunk ≤ `MAX_CHARS`
- sha check: capturing the same conversation twice writes once
- capture off → zero DB writes, `trajectory_ms` NULL
- capture throws → request bytes unchanged, `trajectory_ms` measured, counter bumped
- pruning: inserting past the ceiling drops the oldest conversation whole

Integration:
- a replay run leaves trajectory chunks in the index with `corpus='agentic'` and a
  non-empty `doc_id`
- retrieval with `corpora=('agentic',)` returns only agentic chunks
- the tool-name boost moves a matching chunk ahead of a non-matching one

Gate: every new unit above gets a `run-success-gate.py` entry, following the existing
H-tier pattern.

## 8. Success criteria

- Capture runs on real agentic traffic without measurable latency damage (target: the
  extraction adds < 2 ms per request; the write is off-thread).
- ≥ 70% of held-out corrections are reachable from the failing observation.
- The sweep shows agentic chunks are not worse than code chunks on the agentic cases.

## 9. What this does not claim

It does not claim the model learns. It claims the gateway can record, index, and
retrieve its own tool-use corrections, and that doing so measurably changes what reaches
the prompt. Whether that improves answers is §6.3's job, and §6.3 is allowed to say no.
