"""
One-time migration: transaction_id-only PRIMARY KEY -> composite
(transaction_id, policy_name) PRIMARY KEY, backfilling policy_name='ai_agent'
on every existing row (the only policy that had ever been logged).

SQLite can't ALTER a column into a new composite PK in place, so this does
the standard rebuild: create decisions_new with the target schema (db.SCHEMA,
minus the old table's now-satisfied NOT NULL gaps on the outcome columns —
see note below), copy every old row across with policy_name injected, drop
the old table, rename the new one in.

Note: the OLD schema had outcome_recovered / outcome_recovered_amount /
executed_action / stopping_reason as nullable (step 6 wasn't wired yet when
those rows were written). The NEW schema (db.py) marks them NOT NULL, since
every row logged from now on goes through app/outcome.py. Old rows that
predate outcome simulation get 0/0.0/'unsimulated_legacy_row' placeholders
so the NOT NULL constraint holds — real backfilled outcomes aren't invented
retroactively; the honest move is a visible sentinel, not a fabricated
recovery result. In practice this migration was run before any outcome
data existed, so every row hits this sentinel path — the next batch run
overwrites them with real simulated outcomes anyway (same transaction_id +
policy_name='ai_agent' -> upsert).

Run once: backend/.venv/Scripts/python.exe app/migrate_composite_key.py
Safe to re-run: no-ops if the table already has the composite key.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import DB_PATH, INDEXES, SCHEMA, get_connection  # noqa: E402

LEGACY_STOPPING_REASON = "unsimulated_legacy_row"


def already_migrated(conn) -> bool:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(decisions)")}
    return "policy_name" in cols


def table_exists(conn) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='decisions'"
    ).fetchone()
    return row is not None


def migrate(db_path=DB_PATH) -> dict:
    conn = get_connection(db_path)
    if not table_exists(conn):
        conn.execute(SCHEMA)
        for stmt in INDEXES:
            conn.execute(stmt)
        conn.commit()
        conn.close()
        return {"status": "no existing table — created fresh composite-key schema", "rows_migrated": 0}

    if already_migrated(conn):
        conn.close()
        return {"status": "already migrated — no-op", "rows_migrated": 0}

    old_rows = conn.execute("SELECT * FROM decisions").fetchall()
    n_old = len(old_rows)

    conn.execute("ALTER TABLE decisions RENAME TO decisions_old")
    conn.execute(SCHEMA)
    for stmt in INDEXES:
        conn.execute(stmt)

    for r in old_rows:
        conn.execute(
            """
            INSERT INTO decisions (
                transaction_id, policy_name, batch_id, customer_id,
                event_type, failure_reason, amount, input_event_json,
                predicted_probabilities_json, candidates_ev_ranked_json,
                eligible_actions_json, compliance_state_json,
                selected_action, agent_reason, agent_provider, agent_valid, agent_validation_note,
                gate_status, gate_reason, final_action,
                executed_action, outcome_recovered, outcome_recovered_amount,
                time_to_recovery_hours, stopping_reason,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["transaction_id"], "ai_agent", r["batch_id"], r["customer_id"],
                r["event_type"], r["failure_reason"], r["amount"], r["input_event_json"],
                r["predicted_probabilities_json"], r["candidates_ev_ranked_json"],
                r["eligible_actions_json"], r["compliance_state_json"],
                r["selected_action"], r["agent_reason"], r["agent_provider"], r["agent_valid"], r["agent_validation_note"],
                r["gate_status"], r["gate_reason"], r["final_action"],
                r["executed_action"] or r["selected_action"],
                r["outcome_recovered"] if r["outcome_recovered"] is not None else 0,
                r["outcome_recovered_amount"] if r["outcome_recovered_amount"] is not None else 0.0,
                r["time_to_recovery_hours"],
                r["stopping_reason"] or LEGACY_STOPPING_REASON,
                r["created_at"],
            ),
        )

    conn.execute("DROP TABLE decisions_old")
    conn.commit()
    n_new = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    conn.close()
    return {"status": "migrated", "rows_migrated": n_old, "rows_after": n_new, "policy_name_backfilled": "ai_agent"}


if __name__ == "__main__":
    import json

    print(json.dumps(migrate(), indent=2))
