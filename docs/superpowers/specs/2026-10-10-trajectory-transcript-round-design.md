# Trajectory corpus, round 2 — the operator's own transcripts and the injection decision (2026-10-10)

Parent: `2026-10-08-trajectory-corpus-design.md` (Phase 3-A). Predecessor round:
`AI-Gateway-MLX/reports/trajectory-sweep-2026-10-09/` (PASS, "not worse", capture pilot on).

## 1. The problem, stated as evidence

The 2026-10-09 sweep left one question open: *should the `agentic` corpus be injected by default?* Its
`RESULT.md:120` says what that needs: "a task set on which `cap0` does not already recover". Waiting on the
capture pilot does not provide that.
- **Volume.** After one day the live store has 208 `agentic` chunks:
  - 117 come from conversations, 5 conversations in all;
  - 25 are corrections;
  - 91 are tool schemas.

  At the pilot's rate, a dozen cross-session cases is one to two weeks of active days away.
- **Shape.**
  - Live capture keeps ≤600-character excerpts, while the recovery eval needs full histories
    (`evals/suites/agentic_recovery.py:16`).
  - The grader executes the proposed call through the gateway's MCP tools (`evals/trajectory_sweep.py:214`), so it
    cannot judge `Bash`, `Edit` or `Read`.
  - So even a large captured corpus cannot be measured by the current harness.

Meanwhile the operator's own agent transcripts already hold the data, on this machine:

| source | files | tool calls | failed calls | failure → next same-tool call | fixed within 3 calls |
|---|---|---|---|---|---|
| Claude Code `~/.claude/projects/**/*.jsonl` (2026-08-22 → 2026-10-10) | 1,521 (843 main, 678 subagent) | 70,423 | 2,753 | 2,670 | 2,402 |

How the failures split:
- By month: Aug 202, Sep 1,880, Oct 588 pairs.
- By tool: Bash 2,300, Edit 92, Write 83, Read 80, and others.
- `is_error` marks 62.6%; the rest is detected by the gateway's own text predicate.
- Every record carries `cwd`, `gitBranch` and the model.

These are the operator's real failures on this macOS box, in the operator's repositories, with Claude Code's real
tools. That is exactly the distribution the corpus is for. Public agent datasets are the opposite: Linux
containers, other tool names, other repositories, and third-party text in a store whose write access was
deliberately limited to the operator's own traffic.

## 2. The question, one sentence

> On a failed tool call that Ornith does not fix on its own, does injecting failure→fix examples from the
> operator's *earlier* sessions make its next call the fix more often than injecting the same amount of
> unrelated examples does?

The placebo clause matters. Any injected block perturbs a temperature-0 prompt. On cases selected because the
baseline failed, perturbation alone flips some outcomes, so the comparison is against a block of equal size, not
against nothing.

## 3. What this round is not

- **Not a production change.**
  - Nothing is written to `~/.local/ai-gateway/index.db`, and injection defaults stay as they are.
  - A PASS licenses a separate, ask-first deploy proposal (§9); it does not deploy.
- **Not Codex, in v1.** Its `exec` calls are free-form scripts that the judge in §4.6 cannot compare. Its 476 pairs
  are counted, not used.
- **Not training.** No weights change.
- **Not a claim about other users.** The corpus is one operator's history.

## 4. Design

Six units, each with one job, hermetically testable.

### 4.1 Transcript reader — `evals/transcripts.py`

`sessions(root) -> iterator of Session(id, path, started_at, model, cwd, segments)`. A segment is a list of
canonical OpenAI-shaped messages, the shape `trajectories.events_from_messages` consumes:
- **Tool calls.** An assistant `tool_use` block becomes `{"role":"assistant","tool_calls":[{"id","name","arguments":
  <json>}]}`.
- **Tool results.** A user `tool_result` block becomes `{"role":"tool","tool_call_id","content":<text>}`.
  - With `is_error: true`, the content is prefixed `[tool error] `, as `gateway/compat/anthropic.py:194-195` does.
    Otherwise 392 marked failures with no text marker would be missed.
  - Content shapes: a string as is; `list[text]` joined; `list[image]` becomes `[image]`; `list[tool_reference]`
    becomes the tool names.
- **User and assistant text** are kept as text messages.
- **Compaction.** A `system/compact_boundary` record starts a new segment, because the history before it was
  summarized for the model.
- **Order.** Records are taken in file order. 187 files hold forks; v1 does not reconstruct branches, and a tool
  result always pairs with its call by id, never by position.
- **Subagent files** (`isSidechain`) are sessions of their own.

### 4.2 Failure → fix extraction

- A failure is `trajectories._failed` on the canonical tool message (prefix included), so the rule is the gateway's.
- The fix of a failure is the first later call **of the same tool, within 3 calls**, whose result is not a failure.

Extraction is shared by the importer (§4.3) and the case builder (§4.5), so they cannot disagree about what a
correction is.

### 4.3 Importer — `evals/transcript_import.py --index PATH --before DATE`

For every session that started before the cutoff, in order:
1. `events_from_messages`, then `chunks_from_events`, which applies the per-conversation cap of 24 with
   corrections first.
2. **`trajectories.redact()` on every chunk text.** `ingest()` does not redact; this importer must.
3. `store.replace_file(project="trajectory-local", path="tx-<session id>", …)` with provenance:
   - `corpus: "agentic-local"`
   - `source: "transcript"`
   - `doc_id: "tx-<session id>"`
   - `version_hash: <chunk-set sha>`
   - `generated_by: "transcript-import"`
   - `generated_at: <session start>`

Safeguards:
- **It refuses the live index path** (`~/.local/ai-gateway/index.db`, compared by `realpath`) in this round. It
  writes only to the copy the round uses.
- **A post-import scan** runs every pattern `redact()` knows over every imported chunk. The import fails on any
  match.
- **Its own corpus name.** `agentic-local` keeps the experiment's arms clean, because retrieval filters by corpus
  only (`retrieval.py:108-121`). It also keeps provenance visible: the injected header prints `source`
  (`inject.py:180-184`).
- **Its own project.** `trajectory-local` is outside the live corpus's 20,000-chunk pruning, which counts
  `project='trajectory'` (`trajectories.py:599-628`).

### 4.4 Retrieval

`agentic-local` joins `NAMED_ONLY` (`retrieval.py:105`): it is never read unless a request names it, like `agentic`.
It also gets the rest of `agentic`'s retrieval policy:
- the tool boost (`retrieval.py:139-148`);
- holding the requesting conversation out (`retrieval.py:151-159`).

Without the boost, A1 would test plain BM25 rather than the corpus as production reads a trajectory corpus.
(Amended 2026-10-10 after the final review of the harness; it was "nothing else changes".)

### 4.5 Case builder — `evals/transcript_cases.py --after DATE`

Selection:
- Every failure with a fix (§4.2) in a session that started at or after the cutoff, for the tools `Bash`, `Edit`
  and `Read`.
- Deduplicated by (tool, normalized failing input).
- Sampled deterministically (seed 20261010), stratified by tool, up to **240 cases**.

A case is:
- `id`
- `session`
- `messages`: a fixed system prompt (below) plus the last 12 messages up to and including the failing tool result,
  trimmed from the front to ≤ 16K estimated tokens (`guards.est_tokens`).
- `tools`: fixed Claude Code schemas for `Bash(command)`, `Edit(file_path, old_string, new_string)`,
  `Read(file_path, offset?, limit?)`, `Write(file_path, content)`, `Grep(pattern, path?)` and `Glob(pattern)`.
- `failed`: the failing call.
- `fix`: the fixing call.

The system prompt is: "You are a coding agent on macOS working in the user's repository. Continue the task. When a
tool call fails, make the next call the one that fixes it."

Transcripts record no tool schemas and no system prompt, hence the fixed ones. That is stated in §8.

### 4.6 Judge — `evals/transcript_judge.py` (pure, no execution)

`judge(case, proposed_calls) -> FIX | REPEAT | OTHER | NOCALL`, from the first proposed call:
- **Bash.** Tokens come from `shlex.split` of the command, or whitespace splitting when that fails; similarity is
  Jaccard over tokens.
  - **FIX** if `sim(p, fix) ≥ 0.6` and `sim(p, fix) > sim(p, failed)`.
  - **REPEAT** if `p` equals the failed command after whitespace normalization.
- **Edit.**
  - **FIX** if the same `file_path` as the fix, and `sim(old_string tokens, fix.old_string tokens) ≥ 0.6`.
  - **REPEAT** if `old_string` equals the failed call's.
- **Read.**
  - **FIX** if the same normalized path as the fix: `~` expanded, relative paths joined to the session `cwd`.
  - **REPEAT** if it is identical to the failed call.
- **Any other tool, or the same tool with neither condition:** **OTHER**.
- **No call:** **NOCALL**.

The thresholds are frozen here, before any run. The judge never prints or stores command text in reports, only
outcomes and similarities.

### 4.7 Arms

Retrieval and injection run in process with the gateway's own functions, so the arm cannot drift from production:
- `server._knowledge_query(obj)` for the query, tool hint and `exclude_conv`;
- `context.inject.retrieve(q, corpora, cap, tool_hint, exclude_conv)`;
- `context.inject.block(obj, hits, position=config.KNOWLEDGE_POSITION)`.

These run against the round's copy of the index. The request then goes straight to a scratch `ds4-server` started
from the production Ornith 512K argv (scratch port and KV dir), with:
- `temperature 0`;
- `max_tokens 512`;
- thinking off;
- the case's tools.

| arm | block |
|---|---|
| **A0** `cap0` | none |
| **A1** `local` | `corpora=["agentic-local"]`, cap 2000 |
| **A2** `placebo` | same token size as A1's block for that case. Built from `agentic-local` chunks drawn with a seeded RNG from chunks whose tool differs from the case's tool, rendered with the same `format_block`. A case where A1 injected nothing gets nothing in A2 either |
| **A3** `code` (secondary) | `corpora=["code"]`, cap 2000 |

**Held out.** The case's own session is in the post-cutoff period, so it is not in the corpus. As a second fence,
`exclude_conv` and a `doc_id` filter drop `tx-<case session>`.

### 4.8 Decision rule (frozen before arm 1, in a PRE-REGISTRATION file)

Let H be the hard cases: those where A0 is not FIX.

| outcome | condition |
|---|---|
| **PASS** | all three hold: FIX_A1(H) − FIX_A2(H) ≥ max(3, ⌈0.05·\|H\|⌉); FIX_A1(all) ≥ FIX_A0(all); REPEAT_A1(all) ≤ REPEAT_A0(all) + 2 |
| **FAIL** | otherwise |
| **UNDEFINED** | \|H\| < 30 |
| **VOID** | any of: the index copy changed between arm 1 and arm 4 (row-count and sha census); a hit from the case's own session reached any block; the ds4 binary or argv changed; fewer than 95% of requests completed; the post-import secret scan found a match |

### 4.9 Reports

`reports/trajectory-transcripts-<date>/` holds:
- `PRE-REGISTRATION.md`;
- `cases.json` with ids, tools, session ids and token sizes, never text;
- `arms.json` with outcomes, similarities, block sizes and latencies;
- `RESULT.md`.

No transcript text, command, path or tool output is committed or printed.

## 5. Privacy and trust

- **Same class as the operator's own traffic.** These transcripts are the operator's own sessions on this box, the
  same trust class as permanent-key traffic (`docs/KNOWLEDGE-LAYER.md:141-163`). Nobody else can write them.
- **They stay on this machine.**
  - The round's index copy lives in the session scratchpad and is deleted after the report.
  - Injection runs only on local routes: the hook at `gateway/server.py:1938` sits inside
    `if prov.wants_guards:` (`:1829`), so a cloud route never receives an injected block.
- **Redaction and the scan (§4.3) are gates, not best effort.** Transcripts hold many secret-shaped strings
  (Bearer 325 and key=value 1,338 in tool results alone).
- **No text in the repository.** Reports carry ids, counts and numbers only.

## 6. What "enough data" means now

- **Corpus:** every session before **2026-10-01 00:00** local time, about 2,080 failure→fix pairs, capped at 24
  chunks per session.
- **Cases:** sessions from 2026-10-01 onward, 588 pairs before dedup and the tool filter, sampled to ≤ 240.
- **Volume is no longer the bottleneck.** What remains open is whether the corpus helps, and that is what the rule
  decides.

## 7. Cost

- **Import:** seconds.
- **Arms:** at most 240 cases × 4 arms = 960 requests at ≤ 16K context. On Ornith that is about 3-5 s each, roughly
  1-1.5 h with the gateway's ds4 slot held down. Run it under the quiet-window guard, at night.

## 8. What this round cannot say

- **Most fixes were made by Claude models.** The corpus teaches what fixed the failure, not how Ornith would have
  found it.
- **The judge is a proxy.** A FIX means the proposal resembles the call that worked. OTHER may still have worked,
  so FIX counts are a lower bound in every arm alike.
- **The setting is not production.** Context is truncated, tool schemas and the system prompt are reconstructed, and
  thinking is off.
- **It is one operator's history and one model.**

## 9. What happens after

- **PASS.** A deploy proposal to the operator, asked first:
  1. import sessions into the live index (corpus `agentic-local`), plus a periodic re-import of new sessions;
  2. inject `agentic-local` for **agent requests** (requests that carry tools) on local routes.

  Production injection is off today (`MLX_KNOWLEDGE_INJECT=0`), so how the block is enabled for those requests is
  decided in that proposal, not here. It has its own receipt, Tier L run and one-value rollback.
- **FAIL.** Recorded as a negative finding. The capture pilot continues as is. The corpus stays named-only.
- **UNDEFINED or VOID.** Nothing changes. The reason is recorded.
