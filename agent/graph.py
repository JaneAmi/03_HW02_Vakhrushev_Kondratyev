"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop. Tuned to 2
# (one generate + at most one revise): the baseline eval showed iter_1 == iter_2
# pass rate, so a second revise added no accuracy - we cap at one and reclaim the
# latency. The graph also makes revise terminal (see build_graph), so this is the
# effective bound regardless.
MAX_ITERATIONS = 2

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    evidence: str = ""
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        # SQL queries and the verify JSON are short; cap decode so an occasional
        # verbose generation can't balloon per-call latency (bounds the p95 tail).
        max_tokens=512,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    response = llm().invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
            evidence=state.evidence,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def _parse_verdict(text: str) -> dict:
    """Pull {"ok": bool, "issue": str} out of an LLM reply, defensively.

    The model may wrap the JSON in prose or ```json fences. We grab the first
    balanced-looking object and json.loads it; if that fails we fall back to a
    keyword scan. On a totally unparseable reply we default to ok=True so a
    parser hiccup doesn't burn revise iterations - a genuinely bad result will
    usually still be caught by execution errors / empty rows in the prompt.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return {
                "ok": bool(obj.get("ok", True)),
                "issue": str(obj.get("issue", "") or ""),
            }
        except (json.JSONDecodeError, AttributeError):
            pass
    lowered = text.lower()
    if '"ok": false' in lowered or "'ok': false" in lowered or "not plausible" in lowered:
        return {"ok": False, "issue": text.strip()[:300]}
    return {"ok": True, "issue": ""}


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question."""
    result = state.execution.render() if state.execution is not None else "ERROR: no execution result"
    response = llm().invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            result=result,
        )),
    ])
    verdict = _parse_verdict(response.content)
    return {
        "verify_ok": verdict["ok"],
        "verify_issue": verdict["issue"],
        "history": state.history + [{
            "node": "verify",
            "ok": verdict["ok"],
            "issue": verdict["issue"],
        }],
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt."""
    result = state.execution.render() if state.execution is not None else "ERROR: no execution result"
    response = llm().invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            evidence=state.evidence,
            sql=state.sql,
            result=result,
            issue=state.verify_issue or "the result did not answer the question",
        )),
    ])
    sql = _extract_sql(response.content)
    # Execute inline and terminate (revise is the last step - see build_graph).
    # The baseline eval showed iter_1 == iter_2 pass rate, i.e. a second
    # verify/revise cycle added zero accuracy, so the trailing verify after the
    # first revise was pure latency. We drop it: revise -> execute -> END.
    execution = execute_sql(state.db_id, sql)
    return {
        "sql": sql,
        "execution": execution,
        "iteration": state.iteration + 1,
        # Each attempt's SQL is appended here so the eval can replay per-iteration
        # rows (generate_sql = iter 0, revise = iter 1).
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    # revise executes inline and terminates (single revise, no re-verify) - the
    # per-iteration eval showed a second cycle adds no accuracy, so re-verifying
    # is pure latency. This caps the loop at one revise (3 LLM calls worst case).
    g.add_edge("revise", END)
    return g.compile()


graph = build_graph()
