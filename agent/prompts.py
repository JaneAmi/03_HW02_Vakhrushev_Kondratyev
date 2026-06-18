"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = (
    "You are an expert data analyst who writes SQLite SQL. "
    "Given a database schema and an English question, produce ONE SQLite query "
    "that answers it.\n"
    "Rules:\n"
    "- Use only tables and columns that appear in the schema.\n"
    "- Quote identifiers with double quotes when they contain spaces or are reserved words.\n"
    "- Return exactly the columns the question asks for, nothing extra.\n"
    "- Do not invent data, and do not add LIMIT unless the question implies it.\n"
    "- Output ONLY the SQL inside a single ```sql ... ``` fenced block, no prose."
)

# Available placeholders: {schema}, {question}, {evidence}
GENERATE_SQL_USER = (
    "Database schema:\n"
    "{schema}\n\n"
    "Question: {question}\n\n"
    "Hint (external knowledge - maps question terms to the right columns, coded "
    "values, and derived-metric formulas; use it carefully): {evidence}\n\n"
    "Write the SQLite query that answers the question. When the hint defines a "
    "coded value (e.g. a label or flag) or a metric formula, follow it exactly."
)


VERIFY_SYSTEM = (
    "You are a careful reviewer in a text-to-SQL system. You are given a question, "
    "the SQL that ran, and its execution result. Decide whether the result is a "
    "PLAUSIBLE answer to the question.\n"
    "Default strongly to ok=true. Only return ok=false when you are CONFIDENT the "
    "result is wrong, for one of these concrete reasons:\n"
    "- the SQL errored (the result starts with ERROR);\n"
    "- it returned 0 rows but the question clearly implies at least one row exists;\n"
    "- the result is the wrong SHAPE for the question (e.g. the question asks for a "
    "single count/aggregate but many raw rows came back, or asks for a name but only "
    "an opaque numeric id came back).\n"
    "Do NOT flag a result just because you would have written the query differently, "
    "because column names look unusual, or because you cannot tell whether the exact "
    "values are right - you only see a preview and cannot recompute the answer. "
    "When in doubt, return ok=true.\n"
    'Respond with ONLY a JSON object: {"ok": <true|false>, "issue": "<short reason, empty if ok>"}.'
)

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = (
    "Question: {question}\n\n"
    "SQL that was run:\n{sql}\n\n"
    "Execution result:\n{result}\n\n"
    "Does this result plausibly answer the question? Reply with the JSON object only."
)


REVISE_SYSTEM = (
    "You are an expert SQLite engineer fixing a query that did not answer the "
    "question correctly. You are given the schema, the question, the previous "
    "SQL, its execution result, and a reviewer's complaint.\n"
    "Produce a corrected SQLite query that addresses the complaint.\n"
    "Rules:\n"
    "- Use only tables and columns from the schema.\n"
    "- Fix the specific problem the reviewer raised; do not rewrite blindly.\n"
    "- Quote identifiers with double quotes when needed.\n"
    "- Output ONLY the SQL inside a single ```sql ... ``` fenced block, no prose."
)

# Available placeholders: {schema}, {question}, {evidence}, {sql}, {result}, {issue}
REVISE_USER = (
    "Database schema:\n"
    "{schema}\n\n"
    "Question: {question}\n\n"
    "Hint (external knowledge - coded values, metric formulas): {evidence}\n\n"
    "Previous SQL (incorrect):\n{sql}\n\n"
    "Its execution result:\n{result}\n\n"
    "Reviewer's complaint: {issue}\n\n"
    "Write a corrected SQLite query that answers the question, honoring the hint."
)
