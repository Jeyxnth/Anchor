"""
Run the DECIDE-stage pipeline (policy/pipeline.py) over a batch of events.csv
rows and persist every trace to the audit ledger. Build-order step 7's
write path.

Provider default is explicitly RuleBasedFallbackProvider for batch runs, not
get_provider()'s auto-detected real LLM — calling a live API thousands of
times per batch run is slow, costs quota, and isn't necessary to validate
the ledger/dashboard (the reasoning-string content doesn't affect any
metric). Pass provider=get_provider() explicitly to exercise a real LLM
across a (small) batch instead.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import load_events  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy"))
from pipeline import decide_for_row, load_model  # noqa: E402
from agent import RuleBasedFallbackProvider  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection, init_db  # noqa: E402
from ledger import record_decision_trace  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def run_batch(n: Optional[int] = None, provider=None, batch_id: Optional[str] = None,
              db_path=None) -> dict:
    """Runs the pipeline on the first `n` rows of events.csv (None = all
    4000) and writes each trace to the ledger. Returns a small summary, not
    the full metrics (call ledger.compute_metrics for that)."""
    batch_id = batch_id or f"batch_{uuid.uuid4().hex[:12]}"
    provider = provider or RuleBasedFallbackProvider()

    events_df = load_events(str(DATA_DIR / "events.csv"))
    if n is not None:
        events_df = events_df.head(n)

    model = load_model()
    conn = init_db(db_path) if db_path else init_db()

    n_ok, n_failed = 0, 0
    for _, row in events_df.iterrows():
        try:
            trace = decide_for_row(model, row, provider=provider)
            record_decision_trace(conn, batch_id, row, trace)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - one bad row shouldn't kill the whole batch
            n_failed += 1
            print(f"  [batch {batch_id}] row {row.get('transaction_id')} failed: {exc!r}")
    conn.commit()
    conn.close()

    return {"batch_id": batch_id, "n_cases": n_ok, "n_failed": n_failed, "provider": provider.name}


if __name__ == "__main__":
    import json as _json

    summary = run_batch()
    print(_json.dumps(summary, indent=2))
