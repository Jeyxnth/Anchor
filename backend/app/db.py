"""
Audit ledger persistence — brief §12 / build-order step 7 / MVP deliverable 9.

Plain sqlite3 (brief §17 names SQLite explicitly, "sufficient for MVP") — no
ORM, since the schema is small, fixed, and query needs are simple aggregates.

Schema note — honesty over completeness: several columns exist for fields the
brief's audit spec (§12) requires but that this project hasn't built yet
(the hard compliance GATE — build-order step 5). Those columns are nullable
and stay NULL until that step exists; nothing here fabricates a gate
decision that didn't happen. The execution + outcome simulator (step 6) IS
wired in now (see app/outcome.py) — outcome_recovered etc. are populated for
every row logged via batch.py.

Key note: primary key is (transaction_id, policy_name), not transaction_id
alone. Multiple policies (ai_agent / do_nothing / generic_reminder, for the
baseline experiment — build-order step 8) log a result for the SAME
transaction_id without clobbering each other; re-running the same policy
over the same transaction still replaces that policy's prior row (upsert
semantics), which is the intended "latest decision per (transaction,
policy)" behavior.

Migrated from the original transaction_id-only-PK schema via
migrate_composite_key.py (one-time, already run against ledger.db;
backfilled existing rows with policy_name='ai_agent'). This SCHEMA constant
is what a fresh install gets directly.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "ledger.db"

# A batch run (run_batch()/run_baseline_batch() in app/batch.py) opens one
# connection and doesn't commit until every row in the loop is written —
# so it holds SQLite's write lock for the FULL duration of the run, not
# just per-row. FastAPI dispatches sync `def` endpoints to a thread pool,
# so two batch-writing requests arriving close together (double-click,
# "Run Batch" + "Run Baseline Experiment" overlapping, two browser tabs)
# genuinely execute concurrently — the second one's every row then fights
# the first one's still-open transaction. Root-caused live 2026-09-05: a
# 20-row rule-based batch took ~4 minutes and logged
# `OperationalError('database is locked')` on all 20 rows, because two
# other batches were mid-run at the same time and this connection's bare
# `sqlite3.connect()` (no `timeout=`) only waited SQLite's 5s driver
# default per attempt before giving up — with nothing to retry, every row
# failed outright and the batch's own rows silently never landed (the
# request still returned 200 with n_ok=0 to the caller).
#
# WRITE_LOCK below serializes every ledger-writing call in-process (used by
# app/batch.py and the /recover endpoint in app/main.py) so concurrent
# requests queue cleanly instead of racing SQLite's lock directly. This is
# not a workaround for SQLite being single-writer — SQLite already only
# allows one writer — it's making that constraint explicit and queued at
# the application level instead of surfacing as a silent per-row failure.
WRITE_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    transaction_id              TEXT NOT NULL,
    policy_name                 TEXT NOT NULL,    -- 'ai_agent' | 'do_nothing' | 'generic_reminder' | ...
    batch_id                    TEXT NOT NULL,
    customer_id                 TEXT NOT NULL,

    -- input event snapshot
    event_type                  TEXT NOT NULL,
    failure_reason               TEXT NOT NULL,
    amount                       REAL NOT NULL,
    input_event_json             TEXT NOT NULL,

    -- ML prediction stage (empty dict '{}' for non-ML baseline policies, which don't consult the model)
    predicted_probabilities_json TEXT NOT NULL,   -- {action: P(recovery)} for ALL actions, unfiltered

    -- EV policy stage (empty list '[]' for baseline policies — no EV ranking involved)
    candidates_ev_ranked_json    TEXT NOT NULL,    -- eligibility-filtered, EV-sorted candidate list

    -- compliance ELIGIBILITY filter (pre-LLM candidate restriction; policy/compliance.py) —
    -- computed identically for every policy, including baselines, since compliance never bends
    eligible_actions_json        TEXT NOT NULL,
    compliance_state_json        TEXT NOT NULL,    -- {opted_out, attempts_so_far_24h, last_contact_time}

    -- decision stage (AI agent for policy_name='ai_agent'; a fixed rule for baseline policies —
    -- agent_provider distinguishes which: 'gemini' | 'groq' | 'rule_based_fallback' |
    -- 'baseline_do_nothing' | 'baseline_generic_reminder')
    selected_action               TEXT NOT NULL,    -- chosen action, pre-gate
    agent_reason                  TEXT NOT NULL,
    agent_provider                 TEXT NOT NULL,
    agent_valid                    INTEGER NOT NULL,  -- 1 = action was in allowed list as-returned; 0 = corrected
    agent_validation_note          TEXT,

    -- hard compliance GATE (build-order step 5 — NOT IMPLEMENTED YET; always NULL today)
    gate_status                    TEXT,             -- future: 'approved' | 'blocked'
    gate_reason                    TEXT,
    final_action                   TEXT,             -- future: action actually authorized to execute

    -- execution + outcome simulator (app/outcome.py) — sampled against ground_truth.csv's true
    -- probability for whichever action was selected, same functional form as generate_data.py
    executed_action                 TEXT NOT NULL,    -- = selected_action today (no gate to override it yet)
    outcome_recovered               INTEGER NOT NULL, -- 0/1
    outcome_recovered_amount        REAL NOT NULL,
    time_to_recovery_hours          REAL,             -- NULL if not recovered
    stopping_reason                 TEXT NOT NULL,    -- 'recovered' | 'not_recovered' today (see outcome.py)

    -- timestamps
    created_at                      TEXT NOT NULL,     -- ISO 8601, when this row was written

    PRIMARY KEY (transaction_id, policy_name)
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_decisions_batch ON decisions(batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_policy ON decisions(policy_name)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(selected_action)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_txn ON decisions(transaction_id)",
]


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    # timeout=30: how long THIS connection will busy-wait for a lock before
    # raising OperationalError, instead of the sqlite3 driver's 5s default
    # (see WRITE_LOCK note above for why 5s wasn't enough). Belt-and-
    # suspenders with WRITE_LOCK, not a substitute for it — WRITE_LOCK
    # prevents the contention from happening at all for writers in this
    # process; this timeout is what protects a reader (a /metrics or
    # /audit call) that lands mid-write, or any writer outside this
    # process (e.g. a manual sqlite3 CLI session against ledger.db).
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL instead of the default rollback-journal mode: lets readers
    # (GET /metrics, GET /audit, GET /batch/...) proceed without blocking
    # on an in-progress writer. Doesn't change SQLite's single-writer
    # limit (nothing can) — WRITE_LOCK is what serializes writers.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = get_connection(db_path)
    conn.execute(SCHEMA)
    for stmt in INDEXES:
        conn.execute(stmt)
    conn.commit()
    return conn
