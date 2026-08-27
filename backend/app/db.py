"""
Audit ledger persistence — brief §12 / build-order step 7 / MVP deliverable 9.

Plain sqlite3 (brief §17 names SQLite explicitly, "sufficient for MVP") — no
ORM, since the schema is small, fixed, and query needs are simple aggregates.

Schema note — honesty over completeness: several columns exist for fields the
brief's audit spec (§12) requires but that this project hasn't built yet
(the hard compliance GATE — build-order step 5 — and the execution/outcome
simulators — step 6). Those columns are nullable and stay NULL until those
steps exist; nothing here fabricates a gate decision or an outcome that
didn't happen. Every row currently records real output through the DECIDE
stage (model -> EV ranking -> compliance-ELIGIBILITY filter -> agent
decision), which is everything that exists in the pipeline today.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "ledger.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    transaction_id              TEXT PRIMARY KEY,
    batch_id                    TEXT NOT NULL,
    customer_id                 TEXT NOT NULL,

    -- input event snapshot
    event_type                  TEXT NOT NULL,
    failure_reason               TEXT NOT NULL,
    amount                       REAL NOT NULL,
    input_event_json             TEXT NOT NULL,

    -- ML prediction stage
    predicted_probabilities_json TEXT NOT NULL,   -- {action: P(recovery)} for ALL actions, unfiltered

    -- EV policy stage
    candidates_ev_ranked_json    TEXT NOT NULL,    -- eligibility-filtered, EV-sorted candidate list

    -- compliance ELIGIBILITY filter (pre-LLM candidate restriction; policy/compliance.py)
    eligible_actions_json        TEXT NOT NULL,
    compliance_state_json        TEXT NOT NULL,    -- {opted_out, attempts_so_far_24h, last_contact_time}

    -- AI decision agent stage
    selected_action               TEXT NOT NULL,    -- agent's chosen action, pre-gate
    agent_reason                  TEXT NOT NULL,
    agent_provider                 TEXT NOT NULL,    -- 'gemini' | 'groq' | 'rule_based_fallback'
    agent_valid                    INTEGER NOT NULL,  -- 1 = action was in allowed list as-returned; 0 = corrected
    agent_validation_note          TEXT,

    -- hard compliance GATE (build-order step 5 — NOT IMPLEMENTED YET; always NULL today)
    gate_status                    TEXT,             -- future: 'approved' | 'blocked'
    gate_reason                    TEXT,
    final_action                   TEXT,             -- future: action actually authorized to execute

    -- execution + outcome simulators (build-order step 6 — NOT IMPLEMENTED YET; always NULL today)
    executed_action                 TEXT,
    outcome_recovered               INTEGER,          -- future: 0/1
    outcome_recovered_amount        REAL,
    time_to_recovery_hours          REAL,
    stopping_reason                 TEXT,

    -- timestamps
    created_at                      TEXT NOT NULL      -- ISO 8601, when this row was written
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_decisions_batch ON decisions(batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(selected_action)",
]


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = get_connection(db_path)
    conn.execute(SCHEMA)
    for stmt in INDEXES:
        conn.execute(stmt)
    conn.commit()
    return conn
