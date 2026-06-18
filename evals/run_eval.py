"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _attempts_from_history(history: list[dict]) -> list[str]:
    """Ordered list of SQL attempts: generate_sql (iter 0), revise (iter 1, 2, ...)."""
    return [h["sql"] for h in history if h.get("node") in ("generate_sql", "revise") and "sql" in h]


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question, capturing per-iteration execution accuracy.

    Calls the agent, then replays *each* SQL attempt (from the returned history)
    against the target DB and compares its rows to the gold SQL's rows. This
    gives a correctness flag per iteration, which summarize() carries forward.
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    result = {
        "db_id": db_id,
        "question": question["question"],
        "gold_sql": gold_sql,
        "gold_executable": gold_ok,
        "gold_error": gold_err,
        "agent_ok": False,
        "agent_error": None,
        "final_sql": "",
        "iterations": 0,
        "per_iter": [],       # correctness of attempt 0, 1, 2, ...
        "final_correct": False,
    }

    try:
        resp = httpx.post(
            agent_url,
            json={
                "question": question["question"],
                "db": db_id,
                "evidence": question.get("evidence", ""),
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:  # noqa: BLE001
        result["agent_error"] = f"{type(e).__name__}: {e}"
        return result

    result["agent_ok"] = bool(payload.get("ok", False))
    result["final_sql"] = payload.get("sql", "")
    attempts = _attempts_from_history(payload.get("history", []))
    result["iterations"] = len(attempts)

    per_iter: list[bool] = []
    for sql in attempts:
        _, pred_rows, _ = run_sql(db_id, sql)
        per_iter.append(matches(gold_rows, pred_rows))
    result["per_iter"] = per_iter
    result["final_correct"] = per_iter[-1] if per_iter else False
    return result


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results with per-iteration carry-forward.

    For iteration k, a question that stopped at iteration j < k contributes its
    iteration-j (terminal) result: the agent stopped emitting, so that is what
    would have been served had we polled at k.
    """
    n = len(results)
    max_iters = max((len(r["per_iter"]) for r in results), default=0)

    def correct_at(r: dict, k: int) -> bool:
        per = r["per_iter"]
        if not per:
            return False
        return per[k] if k < len(per) else per[-1]  # carry forward terminal result

    per_iteration_pass_rate = {
        f"iter_{k}": (sum(correct_at(r, k) for r in results) / n if n else 0.0)
        for k in range(max_iters)
    }

    final_correct = sum(1 for r in results if r["final_correct"])
    agent_failures = sum(1 for r in results if not r["agent_ok"])
    gold_unexecutable = sum(1 for r in results if not r["gold_executable"])
    iter_counts = [r["iterations"] for r in results]

    return {
        "n_questions": n,
        "overall_pass_rate": final_correct / n if n else 0.0,
        "n_correct": final_correct,
        "per_iteration_pass_rate": per_iteration_pass_rate,
        "max_iterations_observed": max_iters,
        "avg_iterations": sum(iter_counts) / n if n else 0.0,
        "agent_failures": agent_failures,
        "gold_unexecutable": gold_unexecutable,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
