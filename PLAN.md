# Plan — MLOps Assignment: LLM Inference + Observability

Text-to-SQL self-consistency agent on **Qwen3-30B-A3B-Instruct-2507** (1× H100),
served by vLLM, observed via Prometheus/Grafana + Langfuse, scored with an
execution-accuracy eval, and tuned against an SLO.

> **SLO:** P95 end-to-end agent latency < 5 s, ≥ 10 RPS (1 full agent run/sec)
> over a 5-minute window.

## Strategy: build local, measure on H100

Two distinct stages, deliberately separated:

1. **Local dev (no GPU):** stand up everything end-to-end against **CPU vLLM +
   `Qwen/Qwen3-0.6B`**. CPU vLLM exposes `/metrics`, so we can validate the
   Grafana panels *and* the agent/eval/Langfuse wiring locally. Absolute numbers
   are meaningless here — correctness of the pipeline is the goal.
2. **H100 run:** swap the backend to the real 30B model, then capture **all
   reported numbers and all 8 screenshots**. Phases 1, 6 and the headline eval
   results in 5 are only valid from this run.

Backend is switched purely via `.env` (`VLLM_BASE_URL` / `VLLM_MODEL`), so no
code changes between stages.

## What's already provided (don't rebuild)

- `agent/execution.py`, `agent/schema.py` — SQL execution + schema rendering (complete).
- `agent/graph.py` — graph wiring, `AgentState`, `llm()`, `generate_sql_node` worked example.
- `agent/server.py` — FastAPI `/answer` + Langfuse handler wiring (complete).
- `evals/run_eval.py` — `run_sql`/`canonicalize`/`matches` helpers + `main()` (complete).
- `load_test/driver.py` — async RPS driver with p50/p95/p99 (complete).
- `scripts/load_data.py` — BIRD download + eval/perf split (complete).
- `docker-compose.yml`, `infra/prometheus.yml`, Grafana datasource + 3-panel starter dashboard.

## What we implement (the actual work)

| Area | File | What's left |
|---|---|---|
| Agent | `agent/graph.py` | `verify_node`, `revise_node`, `route_after_verify` |
| Prompts | `agent/prompts.py` | all 6 prompt strings (generate/verify/revise) |
| Eval | `evals/run_eval.py` | `eval_one()`, `summarize()` (per-iteration carry-forward) |
| Dashboard | `infra/grafana/.../serving.json` | latency / throughput / KV-cache panels |
| vLLM cfg | `scripts/start_vllm.sh` | justified flags for the 30B workload |
| Report | `REPORT.md` | full writeup |

---

## Phase-by-phase

### Phase 0 — Setup (local first)
- `cp .env.example .env`; set CPU-vLLM dev block (`VLLM_BASE_URL=http://localhost:8000/v1`, `VLLM_MODEL=Qwen/Qwen3-0.6B`).
- `uv sync` (note: `vllm` wheel install on macOS/CPU may need the CPU install path — verify; fall back to a hosted endpoint only if CPU vLLM won't build).
- `uv run python scripts/load_data.py` → downloads BIRD (~500 MB) → `data/bird/*.sqlite`, `evals/eval_set.jsonl`, `load_test/perf_pool.jsonl`.
- `docker compose up -d`; confirm Prometheus :9090, Grafana :3000 (admin/admin), Langfuse :3001.
- **Done when:** 3 UIs reachable, BIRD data present, `.env` created.

### Phase 1 — vLLM config & justification (H100 for real numbers)
- Start with a sane `start_vllm.sh` for the workload profile (1.5–3K-token prompts, short structured SQL output, 2–3 dependent calls/request) and MoE specifics.
- Candidate levers to justify: `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`/`--max-num-batched-tokens`, quantization (FP8) vs bf16, `--enable-chunked-prefill`, `--kv-cache-dtype`, expert-parallel/`--enable-expert-parallel` for the MoE, `--tensor-parallel-size` (=1 on single H100). Each gets a one-line rationale tying to the SLO.
- Fire 3–5 `eval_set.jsonl` questions manually; confirm sensible SQL.
- **Artifacts:** `screenshots/vllm_manual_query.png`; config + justifications in `REPORT.md`.

### Phase 2 — Observability dashboard (validate on CPU vLLM, capture on H100)
Extend `serving.json` to three categories from vLLM `/metrics`:
- **Latency (percentiles):** `vllm:e2e_request_latency_seconds`, TTFT (`time_to_first_token`), TPOT/inter-token, plus queue/prefill/decode breakdown to answer "*where* in the lifecycle".
- **Throughput:** `generation_tokens_total` rate, prompt tokens/s, `request_success_total` rate, `num_requests_running` / `num_requests_waiting` (queue depth).
- **KV cache:** `gpu_cache_usage_perc` (+ preemptions / evictions) for concurrency headroom.
- Every panel must visibly react under a request burst.
- **Artifacts:** `screenshots/grafana_serving.png`; committed `serving.json`.

### Phase 3 — Agent (build local, final prompt-tune on H100)
- `prompts.py`: `GENERATE_SQL_*` (schema+question → SQL only), `VERIFY_*` (asks "does this plausibly answer the question?" → strict `{"ok": bool, "issue": str}` JSON), `REVISE_*` (failing SQL + execution result + issue → fixed SQL).
- `graph.py`:
  - `verify_node` — call `llm()` with verify prompts, feed `state.execution.render()`, parse JSON defensively → `{verify_ok, verify_issue}`.
  - `revise_node` — same shape as `generate_sql_node`, include prior SQL/result/issue, bump `iteration`.
  - `route_after_verify` — `"end"` if `verify_ok` or `iteration >= MAX_ITERATIONS`, else `"revise"`.
- Verify heuristics to catch: SQL errored, 0 rows when question implies rows, columns clearly don't answer.
- Run agent server `uv run uvicorn agent.server:app --port 8001`; test 5 questions; **confirm ≥1 triggers a revise**.

### Phase 4 — Langfuse tracing (backend-agnostic)
- Sign up at :3001, create project, copy public/secret keys into `.env`.
- Handler is already wired in `server.py` (`langfuse.langchain.CallbackHandler`); just provide keys.
- Fire 10+ questions; tag traces with metadata (e.g. `phase`, `run_id`) via the `/answer` `tags` field for Phase-6 filtering.
- **Artifacts:** `screenshots/langfuse_trace.png` (generate_sql→verify→revise waterfall), `screenshots/langfuse_tags.png`.

### Phase 5 — Eval framework (wiring local, numbers on H100)
- `eval_one()`: POST question to agent → run agent SQL + gold SQL via `run_sql` → `matches()`; capture iterations taken and **per-iteration correctness** (needs history from `/answer` — confirm `history` carries enough; extend `AgentState.history`/server response if per-iteration SQL isn't exposed).
- `summarize()`: overall pass rate + pass rate@iter 0/1/2 with **carry-forward** (a question that stopped at iter j counts its iter-j result for all k>j).
- Run 30 questions while watching Grafana → `results/eval_baseline.json`.
- **Artifacts:** `screenshots/grafana_eval_run.png`; read on whether the loop earns its keep (iter0 vs iter-max pass rate).

> Note: `eval_one`/`summarize` need per-iteration SQL. The carrier already
> exists end-to-end — `AgentState.history` (populated by `generate_sql_node`)
> is returned by the server. The only change is **one line in `revise_node`'s
> return** appending its attempt to `history`; `eval_one` then replays each
> attempt's SQL against the gold DB. No server change needed.

### Phase 6 — SLO diagnosis & iteration (H100 only) — heaviest weight (25%)
- Baseline: `uv run python load_test/driver.py --rps 10 --duration 300`, watch Grafana.
- Loop: read dashboard → which metric moves first (queue depth? KV cache? TTFT?) → form grounded hypothesis → change **one** thing → confirm targeted metric moved → check e2e latency followed. 3–4 iterations.
- Each iteration: one-line log *"saw X → hypothesized Y → changed Z → result W"* + a Grafana screenshot.
- Re-run eval → `results/eval_after_tuning.json`; confirm quality didn't regress.
- **Artifacts:** `screenshots/grafana_before.png`, `grafana_after.png`; iteration log + honest SLO verdict in `REPORT.md`.

### Phase 7 — Report
`REPORT.md` (≤3 pages): serving flags+justifications; baseline eval (overall + per-iter + commentary); SLO baseline-vs-target + iteration log + final numbers; one paragraph on agent-loop value (cite per-iter pass rate); specific future work.

---

## Deliverables checklist
- [ ] `REPORT.md`
- [ ] `infra/grafana/provisioning/dashboards/serving.json` (latency/throughput/KV panels)
- [ ] `agent/graph.py`, `agent/prompts.py`
- [ ] `evals/run_eval.py`
- [ ] `results/eval_baseline.json`, `results/eval_after_tuning.json`
- [ ] 8 screenshots: `vllm_manual_query`, `grafana_serving`, `langfuse_trace`, `langfuse_tags`, `grafana_eval_run`, `grafana_before`, `grafana_after`
- [ ] `scripts/start_vllm.sh` with tuned flags

> **gitignore:** fixed — `.gitignore` now whitelists the named deliverable JSONs
> and the 8 screenshots while still ignoring transient files (`load_test.json`,
> stray images). They commit normally; no `git add -f` needed.

## Execution order
1. Phase 0 + 3 + 4 + 5-wiring locally on CPU vLLM (Qwen3-0.6B) — get the whole pipeline green.
2. Validate Phase 2 dashboard reacts under a local burst.
3. Book H100 → re-point `.env` to 30B → Phases 1, 2 (capture), 5 (real numbers), 6 (tune), capture all screenshots.
4. Phase 7 writeup.
