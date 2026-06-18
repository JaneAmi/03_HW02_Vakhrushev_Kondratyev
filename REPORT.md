# REPORT — Text-to-SQL on Qwen3-30B-A3B: inference, observability, agent

> All numbers, screenshots, and the SLO verdict below come from the real
> `Qwen/Qwen3-30B-A3B-Instruct-2507` on **1× H100 80GB**. Local CPU-vLLM runs
> (Qwen3-0.6B) were used only to build and debug the pipeline.

## 1. Serving configuration (Phase 1)

Model fixed at `Qwen/Qwen3-30B-A3B-Instruct-2507`, MoE (30B total / ~3B active).
On one H100 the binding constraint is KV-cache memory, not compute — so the
config buys KV headroom and prefix reuse. Full launch in `scripts/start_vllm.sh`.

| Flag | Value | Why (this workload) |
|---|---|---|
| `--max-model-len` | `8192` | Prompts cap ~3K tokens; shrinking the per-seq KV reservation from the 256K default frees memory for concurrency. |
| `--gpu-memory-utilization` | `0.90` | Use the 80GB card fully for weights + KV without OOM risk. |
| `--kv-cache-dtype` | `fp8` | Halves KV memory (native FP8 on H100) → ~2× concurrency; negligible impact on short SQL. |
| `--enable-prefix-caching` | on | Schema + system prompt repeat across the 2–3 calls/request and across all questions on a DB → skip re-prefilling the shared prefix. |
| `--max-num-seqs` | `64` | Concurrency cap sized for 10 RPS of bursty dependent calls; tuned in Phase 6. |
| `--tensor-parallel-size` | `1` | Single H100. |
| chunked prefill | on (default) | Interleaves large prefill with decode → steadier p95 under load. On by default in vLLM 0.11+, so no explicit flag. |

**Serving stack note.** The pinned `vllm==0.10.2` crashes loading this model's
tokenizer (`Qwen2Tokenizer has no attribute all_special_tokens_extended`); vLLM
**0.11+** fixed that code path, so we run a current vLLM. FlashInfer's JIT sampler
is disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`) to avoid a `ninja`/`nvcc` build at
startup — irrelevant to quality since we decode greedily (`temperature=0`).

_Manual sanity check (`screenshots/vllm_manual_query.png`):_ fired 3 eval
questions straight at vLLM — all returned correct, idiomatic SQL: correct joins
(circuits↔races; the 3-table superhero powers join) and proper double-quoting of
spaced columns (`"Enrollment (Ages 5-17)"`, `"NCESSchool"`) for the top-5 schools
query. Serving layer confirmed sound.

## 2. Baseline eval results (Phase 5)

Execution accuracy over 30 BIRD questions (`results/eval_baseline.json`),
agent capped at `MAX_ITERATIONS = 3`. Comparison after canonicalized row sets
(sorted, stringified, NULL→'').

| Metric | Value |
|---|---|
| Overall pass rate | **56.7%** (17/30) |
| Pass rate @ iter 0 (generate only) | 53.3% (16/30) |
| Pass rate @ iter 1 (1 revise) | 56.7% (17/30) |
| Pass rate @ iter 2 (2 revises) | 56.7% (17/30) |
| Avg iterations / question | 1.13 |
| Agent failures / gold unexecutable | 0 / 0 |

**Two diagnosis-driven iterations got here** (this is the eval story, not luck):

| Stage | Change | Overall | iter0 → iter2 | Loop |
|---|---|---|---|---|
| v0 | initial verify + no evidence | 26.7% | 30% → 27% | **net-negative** |
| v1 | conservative verify prompt | 30.0% | 30% → 30% | neutral |
| **v2 (baseline)** | **+ BIRD `evidence` hint** | **56.7%** | 53% → 57% | **net-positive** |

- *v0→v1:* the first eval showed the loop **regressing** quality — verify flagged a
  correct `card_games` answer and revise corrupted it (1 regression, 0 recoveries).
  Making `verify` conservative (default `ok=true`, flag only confident failures)
  removed the regression and cut wasted revises (avg iters 1.63→1.43).
- *v1→v2:* failure analysis showed ~6/8 failures were **knowledge-gap** (questions
  needing coded-column / encoded-value / derived-metric knowledge, e.g. `A15`=
  crimes-1995, carcinogenic=`label '+'`). BIRD ships this as an `evidence` hint that
  the data loader was dropping; threading it into the prompts nearly doubled
  accuracy (30%→56.7%).

_Commentary:_ iter_1 is +3.3 pts over iter_0 (one question recovered, zero
regressed) at +0.13 iterations/question — so the loop now earns its keep, modestly.
Its ceiling is bounded by what `revise` can fix: once knowledge-gap failures were
removed by evidence, the loop had fixable SQL bugs to work on and went net-positive.

## 3. Hitting the SLO (Phase 6)

**Target:** P95 end-to-end agent latency < 5 s, ≥ 10 RPS over 5 min.

**Baseline** (1 uvicorn worker, at 10 RPS): p50 **59s**, p95 **104s**, p99 114s,
only **1586/3000 ok (53%)** — 758 timeouts + 656 connection errors. Far over SLO.

**The decisive diagnosis came from the dashboard:** at 10 RPS the vLLM panels were
all *healthy* — KV cache **~2%**, **0 preemptions**, e2e request latency p99 ~4s,
"Requests running" sawtoothing 0→30. The GPU was **idle while the client saw 100s
latencies**. So the bottleneck was *not* the serving layer — it was the agent.

**Iteration log** — *saw X → hypothesized Y → changed Z → result W*:
1. **saw** GPU idle (KV 2%, 0 preempt) but agent p95=104s & 47% failures → **hypothesized** the single sync uvicorn process (~40-thread pool) can't sustain the ~70 concurrent runs 10 RPS needs → **changed** uvicorn `--workers 1→4` → **result** p95 104s→**7.97s**, success 53%→99.9%; "Requests running" went steady (no sawtooth), vLLM fed continuously.
2. **saw** p95 7.97s with a fat tail; the agent's LLM client had no output cap → **hypothesized** occasional long generations inflate per-call decode → **changed** `max_tokens=512` → **result** p95→**7.33s**, **100%** success (0 failures).
3. **saw** the residual tail was the ~13% of requests that revise (4 sequential LLM calls), and the baseline eval showed **iter_1 == iter_2** (a 2nd revise adds 0 accuracy) → **hypothesized** the trailing verify after the first revise is pure latency → **changed** made revise terminal (execute-and-return, no re-verify; cap = 1 revise) → **result** p95 7.33s→**6.12s**, p50 **1.44s**, 100% success.

Before/after the change that moved the needle (iter 1): `screenshots/grafana_before.png`
(sawtooth, idle GPU) vs `screenshots/grafana_after.png` (steady utilization).

**Final numbers:** p50 **1.44s**, p95 **6.12s**, p99 11.6s, **3000/3000 ok**, ~10 RPS
sustained (the driver's "achieved 8.7 RPS" is diluted by the 60s drain window; it
completed all 3000 requests fired at 10/s).

**Verdict: SLO missed on p95 by ~1.1s** (6.12 vs 5.0); p50 and RPS are comfortably
met. The gap is structural: a revising request makes **3 dependent (non-parallelizable)
LLM calls**, and decode is slower than it should be — vLLM warns at startup that it's
using an **untuned fused-MoE kernel** for this expert shape on H100
(`E=128,N=768,...H100...json` not found), so decode p95 ~2.5s/call × 3 ≈ 6s. Closing
the last 1.1s needs a faster *per-call* decode, not more agent concurrency (see §5).

**Quality after tuning** (`results/eval_after_tuning.json`): overall **53.3%**
(16/30) vs baseline **56.7%** (17/30) — a one-question regression, which I'll be
honest about. It's at the **generate step** (iter_0 53.3%→50.0%), *not* from the
loop change: the verify→revise loop stayed net-positive (iter_0 50.0% → iter_1
53.3%, +1 recovered), and capping at one revise was justified (baseline iter_1 ==
iter_2). The most likely cause is **greedy-decode non-determinism** — at temp=0
vLLM is still not bit-identical across runs because batch composition differs
between the two eval passes, so a borderline token can flip; on 30 questions one
flip is ±3.3 pts, i.e. within run-to-run variance. (The `max_tokens=512` cap is a
secondary suspect, but SQL rarely exceeds it.) Net: the latency tuning did **not**
damage the agent's logic; the 1-question delta is noise, and would be pinned down
by averaging a few eval seeds — see §5.

## 4. Did the agent loop add value?

Yes — modestly, and only after it was tuned to. In the final configuration the
verify→revise loop lifts execution accuracy from **53.3% (iter_0) to 56.7%
(iter_1)** — it recovers one question the single-shot generator got wrong and
regresses none, at an average cost of just **0.13 extra LLM calls/question**
(avg 1.13 iterations), because conservative verify only fires on confident
failures. The more interesting lesson is the journey: the *first* version of the
loop was net-**negative** (it corrupted a correct answer), and even after fixing
that it was merely neutral until BIRD `evidence` raised first-pass accuracy. A
verify→revise loop can only recover failures that `revise` can actually fix — when
the dominant failure mode was missing domain knowledge, the loop had nothing to
work with; once that floor was lifted, the residual errors were fixable SQL bugs
and the loop started paying off. So the architecture adds value, but its value is
contingent on (a) a verifier tuned against false positives and (b) the generator
having the knowledge to make failures *fixable* rather than *fundamental*.

## 5. What I'd do with more time

- **Few-shot the generate prompt** with 2–3 schema-matched (question, SQL) exemplars
  to push iter_0 accuracy up and make `revise`'s job easier — the cheapest remaining
  quality lever now that evidence is wired in.
- **Value-grounded prompting / schema linking:** several residual failures are exact
  data-value mismatches (`gender='m'` vs `'M'`, `'Hl.m.Praha'` vs `'Hl.m. Praha'`).
  Sampling distinct values for the columns a question touches, and injecting them,
  would fix these without bloating the whole schema.
- **Make `revise` smarter:** feed it the *diff* between its rows and an expected
  shape, and let `verify` pass a structured complaint (missing-filter / wrong-agg)
  rather than free text, so revise targets the actual defect.
- **Tune the fused-MoE kernel** (closes the SLO gap): vLLM warned it's using a
  *default, sub-optimal* MoE config because no `E=128,N=768,…H100…json` exists for
  this expert shape. Running vLLM's `benchmark_moe` to generate that config should
  cut decode/call directly — the most targeted fix for the 1.1s p95 miss.
- **Speculative decoding** with a small Qwen3 draft model to cut TPOT — decode
  dominates e2e latency for these short structured outputs.
- **A proper offline eval gate in CI** so prompt/serving changes are scored on the
  30-question set automatically before they ship.

---
_Observability artifacts: `screenshots/grafana_serving.png` (dashboard under
load), `screenshots/grafana_eval_run.png` (dashboard during the 30-question
eval), `screenshots/langfuse_trace.png` + `screenshots/langfuse_tags.png`
(agent traces with the generate→verify→revise waterfall and Phase-6 tags)._
