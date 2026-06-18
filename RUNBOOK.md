# RUNBOOK — turnkey command sequence

Exact steps to run the pipeline on a Linux VM (CPU-vLLM dev box, then the H100).
The agent/eval/dashboard code is already implemented; this is the operational
sequence. Pairs with `PLAN.md` (what & why) and `REPORT.md` (results).

> Code already validated offline (agent loop + eval logic). What's left is
> standing up the services and capturing real numbers/screenshots.

---

## 0. Connect & forward ports (from your laptop)

```bash
ssh -L 3000:localhost:3000 \   # Grafana
    -L 9090:localhost:9090 \   # Prometheus
    -L 3001:localhost:3001 \   # Langfuse
    -L 8000:localhost:8000 \   # vLLM
    -L 8001:localhost:8001 \   # agent server
    <user>@<vm-host>
```
(Or use VSCode/Cursor Remote-SSH → Ports panel → forward all five.)

## 1. One-time setup (on the VM)

```bash
# uv (if missing)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

cd mlops_assignment
uv sync                                   # installs vLLM + agent deps (Linux/CUDA wheel)
cp .env.example .env                      # then edit per stage below
uv run python scripts/load_data.py        # ~500MB BIRD -> data/bird/, eval_set.jsonl, perf_pool.jsonl
docker compose up -d                      # prometheus :9090, grafana :3000, langfuse :3001

# sanity
curl -s localhost:9090/-/healthy && echo                    # Prometheus
curl -s -o /dev/null -w "grafana %{http_code}\n" localhost:3000
curl -s -o /dev/null -w "langfuse %{http_code}\n" localhost:3001
```

`.gitignore` already whitelists the graded deliverables, so generated
`results/*.json` and `screenshots/*.png` commit normally — no `git add -f`.

---

## STAGE A — local dev validation (CPU vLLM, small model)

Goal: prove the whole pipeline is green and the Grafana panels react. Numbers
here are throwaway. `.env`:

```ini
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen3-0.6B
OPENAI_API_KEY=not-needed
```

```bash
# CPU vLLM with the small stand-in model (exposes /metrics for Grafana)
# See https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html
uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --device cpu &

# agent server (terminal 2)
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

# smoke one question (terminal 3); confirm sensible SQL comes back
DB=$(uv run python -c "from agent.schema import available_dbs; print(available_dbs()[0])")
curl -s -X POST localhost:8001/answer -H 'Content-Type: application/json' \
  -d "{\"question\":\"How many rows are in the largest table?\",\"db\":\"$DB\"}" | python -m json.tool

# eval harness end-to-end (validates wiring, not real accuracy)
uv run python evals/run_eval.py --out results/_dev_eval.json

# fire a burst and watch every Grafana panel move (Phase 2 check)
uv run python load_test/driver.py --rps 4 --duration 60 --out results/_dev_load.json
```

Open Grafana (`localhost:3000`, admin/admin) → **vLLM serving** dashboard;
confirm latency / throughput / queue / KV panels all react. Delete the `_dev_*`
files afterward — they are not deliverables.

---

## STAGE B — H100 run (the real numbers + all screenshots)

Switch `.env` back to the real model (default block):

```ini
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
HF_TOKEN=<your hf token>
# OPENAI_API_KEY left as-is; vLLM ignores it
```

### B1 — serve + manual queries (Phase 1)
```bash
bash scripts/start_vllm.sh          # tuned flags live here; wait for "Application startup complete"
# fire 3-5 eval questions manually:
uv run python - <<'PY'
import json, httpx
from pathlib import Path
qs = [json.loads(l) for l in Path("evals/eval_set.jsonl").read_text().splitlines()[:5]]
for q in qs:
    r = httpx.post("http://localhost:8001/answer",
                   json={"question": q["question"], "db": q["db_id"]}, timeout=120).json()
    print(q["db_id"], "|", q["question"][:60], "->", r["sql"][:80], "| ok:", r["ok"])
PY
```
📸 **`screenshots/vllm_manual_query.png`** — vLLM serving + a manual query returning SQL.

### B2 — dashboard under load (Phase 2)
```bash
uv run python load_test/driver.py --rps 8 --duration 120   # background traffic while you grab the shot
```
📸 **`screenshots/grafana_serving.png`** — full dashboard, panels reacting.

### B3 — agent + Langfuse (Phases 3–4)
1. Langfuse UI (`localhost:3001`) → sign up → create project → copy public/secret keys into `.env` → restart the agent server.
2. Fire 10+ questions (the B1 loop, bumped to 10, with `"tags": {"phase":"baseline"}` in the payload).
3. In Langfuse, open one trace with a revise.
   - 📸 **`screenshots/langfuse_trace.png`** — generate_sql → verify → revise waterfall.
   - 📸 **`screenshots/langfuse_tags.png`** — trace list with your metadata tags.

### B4 — baseline eval (Phase 5)
```bash
uv run python evals/run_eval.py --out results/eval_baseline.json   # 30 Q, watch Grafana
```
📸 **`screenshots/grafana_eval_run.png`** — dashboard during the eval.
→ fill REPORT §2 (overall + per-iteration pass rate; does the loop earn its keep?).

### B5 — SLO load test + tuning loop (Phase 6, 25% weight)
```bash
uv run python load_test/driver.py --rps 10 --duration 300 --out results/load_test.json
```
- Read the dashboard → which metric moves first (queue depth? KV usage? TTFT? preemptions?).
- Change **one** vLLM flag (see the "Phase 6 levers" block in `scripts/start_vllm.sh`), restart vLLM, re-run.
- Log each iteration in REPORT §3: *saw X → hypothesized Y → changed Z → result W*.
- 📸 **`screenshots/grafana_before.png`** / **`grafana_after.png`** around the change that moved the needle.
- Re-run eval on the final config:
```bash
uv run python evals/run_eval.py --out results/eval_after_tuning.json
```

### B6 — write-up (Phase 7)
Fill every `<TODO>` in `REPORT.md`; confirm all 8 screenshots + both eval JSONs present.

---

## Final deliverables check
```bash
ls -1 screenshots/*.png            # expect 7 files (grafana_before+after = 2)
ls -1 results/eval_baseline.json results/eval_after_tuning.json
git add -A && git commit -m "mlops: H100 results, screenshots, final report"
```

## Teardown
```bash
docker compose down            # add -v to also drop volumes (langfuse/grafana data)
# stop the vLLM and uvicorn processes (Ctrl-C or kill the backgrounded PIDs)
```
