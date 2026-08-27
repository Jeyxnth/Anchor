import { useEffect, useState } from "react";
import { api } from "./api";
import "./App.css";

const ACTION_LABELS = {
  retry_link: "Retry Link",
  reminder: "Reminder",
  discount_offer: "Discount Offer",
  escalate_to_human: "Escalate to Human",
  no_action: "No Action",
};

function formatINR(amount) {
  if (amount === null || amount === undefined) return "—";
  return `₹${amount.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

function PendingNote({ children }) {
  return <span className="pending-note" title={children}>pending ⓘ</span>;
}

function KpiCard({ label, value, pendingNote }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value ${value === null ? "kpi-value--pending" : ""}`}>
        {value === null ? "—" : value}
      </div>
      {value === null && pendingNote && <PendingNote>{pendingNote}</PendingNote>}
    </div>
  );
}

function InterventionBreakdown({ data }) {
  const entries = Object.entries(data || {});
  const maxAmount = Math.max(1, ...entries.map(([, v]) => v.amount_at_risk));
  return (
    <div className="panel">
      <h3>Recovery by Intervention</h3>
      {entries.length === 0 && <p className="muted">No decisions logged yet.</p>}
      <div className="intervention-list">
        {entries.map(([action, v]) => (
          <div className="intervention-row" key={action}>
            <div className="intervention-name">{ACTION_LABELS[action] || action}</div>
            <div className="intervention-bar-track">
              <div
                className="intervention-bar-fill"
                style={{ width: `${(v.amount_at_risk / maxAmount) * 100}%` }}
              />
            </div>
            <div className="intervention-stats">
              <span>{v.n_selected} case{v.n_selected === 1 ? "" : "s"}</span>
              <span>{formatINR(v.amount_at_risk)} at risk</span>
              <span>
                mean P(recover) ={" "}
                {v.mean_predicted_recovery_probability !== null
                  ? `${(v.mean_predicted_recovery_probability * 100).toFixed(1)}%`
                  : "—"}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CompliancePanel({ compliance }) {
  if (!compliance) return null;
  return (
    <div className="panel">
      <h3>Compliance Panel</h3>
      <div className="compliance-grid">
        <div className="compliance-stat">
          <div className="compliance-stat-value">{compliance.cases_with_restricted_eligibility}</div>
          <div className="compliance-stat-label">Cases with restricted eligibility</div>
        </div>
        <div className="compliance-stat">
          <div className="compliance-stat-value">{compliance.opted_out_respected}</div>
          <div className="compliance-stat-label">Opt-outs respected</div>
        </div>
        <div className="compliance-stat">
          <div className="compliance-stat-value">{compliance.contact_cap_restricted}</div>
          <div className="compliance-stat-label">Contact-limit restricted</div>
        </div>
        <div className="compliance-stat">
          <div className="compliance-stat-value">
            {compliance.quiet_hour_violations === null ? <PendingNote>{compliance._note}</PendingNote> : compliance.quiet_hour_violations}
          </div>
          <div className="compliance-stat-label">Quiet-hour violations</div>
        </div>
        <div className="compliance-stat compliance-stat--highlight">
          <div className="compliance-stat-value">{compliance.target_compliance_violations}</div>
          <div className="compliance-stat-label">Target compliance violations</div>
        </div>
      </div>
    </div>
  );
}

function DecisionsTable({ decisions, onSelect }) {
  return (
    <div className="panel">
      <h3>Recent Decisions {decisions.length > 0 && <span className="muted">(showing {decisions.length})</span>}</h3>
      {decisions.length === 0 ? (
        <p className="muted">No decisions logged yet — run a batch above.</p>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Transaction</th>
                <th>Customer</th>
                <th>Amount</th>
                <th>Failure Reason</th>
                <th>Selected Action</th>
                <th>P(recover)</th>
                <th>Provider</th>
                <th>Valid</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map((d) => (
                <tr key={d.transaction_id} onClick={() => onSelect(d.transaction_id)} className="clickable-row">
                  <td>{d.transaction_id}</td>
                  <td>{d.customer_id}</td>
                  <td>{formatINR(d.amount)}</td>
                  <td>{d.failure_reason}</td>
                  <td>{ACTION_LABELS[d.selected_action] || d.selected_action}</td>
                  <td>{(d.predicted_probabilities[d.selected_action] * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`provider-badge provider-badge--${d.agent_provider}`}>{d.agent_provider}</span>
                  </td>
                  <td>{d.agent_valid ? "✓" : "corrected"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function TraceModal({ trace, onClose }) {
  if (!trace) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{trace.transaction_id} — full decision trace</h3>
          <button onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          <section>
            <h4>Context</h4>
            <p>
              {trace.customer_id} · {formatINR(trace.amount)} · {trace.event_type} / {trace.failure_reason}
            </p>
          </section>
          <section>
            <h4>Predicted recovery probability per action</h4>
            <ul className="plain-list">
              {Object.entries(trace.predicted_probabilities).map(([a, p]) => (
                <li key={a}>
                  {ACTION_LABELS[a] || a}: {(p * 100).toFixed(1)}%
                </li>
              ))}
            </ul>
          </section>
          <section>
            <h4>Eligible actions (compliance filter)</h4>
            <p>{trace.eligible_actions.map((a) => ACTION_LABELS[a] || a).join(", ")}</p>
          </section>
          <section>
            <h4>EV-ranked candidates</h4>
            <table className="compact-table">
              <thead>
                <tr>
                  <th>Action</th>
                  <th>P(recover)</th>
                  <th>Cost</th>
                  <th>Discount cost</th>
                  <th>EV</th>
                </tr>
              </thead>
              <tbody>
                {trace.candidates_ev_ranked.map((c) => (
                  <tr key={c.action} className={c.action === trace.selected_action ? "row-selected" : ""}>
                    <td>{ACTION_LABELS[c.action] || c.action}</td>
                    <td>{(c.predicted_probability * 100).toFixed(1)}%</td>
                    <td>{formatINR(c.intervention_cost)}</td>
                    <td>{formatINR(c.discount_cost)}</td>
                    <td>{formatINR(c.expected_value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <section>
            <h4>Agent decision</h4>
            <p>
              <strong>{ACTION_LABELS[trace.selected_action] || trace.selected_action}</strong> via{" "}
              <span className={`provider-badge provider-badge--${trace.agent_provider}`}>{trace.agent_provider}</span>
            </p>
            <p className="muted">{trace.agent_reason}</p>
            {!trace.agent_valid && (
              <p className="warning-note">⚠ {trace.agent_validation_note}</p>
            )}
          </section>
          <section>
            <h4>Downstream stages (not yet implemented)</h4>
            <p className="muted">
              Hard compliance gate: {trace.gate_status ?? "pending (build-order step 5)"} · Executed action:{" "}
              {trace.executed_action ?? "pending (step 6)"} · Outcome:{" "}
              {trace.outcome_recovered === null || trace.outcome_recovered === undefined
                ? "pending (step 6)"
                : trace.outcome_recovered
                ? `recovered ${formatINR(trace.outcome_recovered_amount)}`
                : "not recovered"}
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [metrics, setMetrics] = useState(null);
  const [decisions, setDecisions] = useState([]);
  const [batchId, setBatchId] = useState(null);
  const [nRows, setNRows] = useState(200);
  const [useRealLlm, setUseRealLlm] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [apiUp, setApiUp] = useState(null);

  useEffect(() => {
    api.health().then(() => setApiUp(true)).catch(() => setApiUp(false));
  }, []);

  async function refresh(id) {
    const [m, b] = await Promise.all([api.metrics(id), api.batchDecisions(id)]);
    setMetrics(m);
    setDecisions(b.decisions);
  }

  async function handleRunBatch() {
    setRunning(true);
    setError(null);
    try {
      const summary = await api.runBatch(nRows || null, useRealLlm);
      setBatchId(summary.batch_id);
      await refresh(summary.batch_id);
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  async function handleSelectRow(transactionId) {
    try {
      const trace = await api.audit(transactionId);
      setSelectedTrace(trace);
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Revenue Recovery Control Center</h1>
          <p className="muted">
            API: {apiUp === null ? "checking…" : apiUp ? "connected" : "unreachable — is uvicorn running on :8000?"}
            {batchId && <> · batch: {batchId}</>}
          </p>
        </div>
        <div className="batch-controls">
          <label>
            rows
            <input
              type="number"
              min="1"
              max="4000"
              value={nRows}
              onChange={(e) => setNRows(Number(e.target.value))}
            />
          </label>
          <label className="checkbox-label">
            <input type="checkbox" checked={useRealLlm} onChange={(e) => setUseRealLlm(e.target.checked)} />
            use real LLM (slow — one API call per row)
          </label>
          <button onClick={handleRunBatch} disabled={running}>
            {running ? "Running…" : "Run Batch"}
          </button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="kpi-row">
        <KpiCard label="Revenue at Risk" value={metrics ? formatINR(metrics.revenue_at_risk) : null} />
        <KpiCard
          label="Revenue Recovered"
          value={metrics && metrics.revenue_recovered !== null ? formatINR(metrics.revenue_recovered) : null}
          pendingNote={metrics?._pending_note}
        />
        <KpiCard
          label="Recovery Rate"
          value={metrics && metrics.recovery_rate !== null ? metrics.recovery_rate : null}
          pendingNote={metrics?._pending_note}
        />
        <KpiCard
          label="Incremental Recovery vs. Baseline"
          value={metrics && metrics.incremental_recovery_vs_baseline !== null ? metrics.incremental_recovery_vs_baseline : null}
          pendingNote={metrics?._pending_note}
        />
      </section>

      {metrics && (
        <p className="scope-note">
          {metrics._pending_note}
        </p>
      )}

      <div className="panel-grid">
        <InterventionBreakdown data={metrics?.recovery_by_intervention} />
        <CompliancePanel compliance={metrics?.compliance} />
      </div>

      <div className="panel">
        <h3>Stopping-Reason Breakdown</h3>
        <p className="muted">
          <PendingNote>Requires the execution/outcome simulator (build-order step 6), not implemented yet.</PendingNote>
        </p>
      </div>

      <DecisionsTable decisions={decisions} onSelect={handleSelectRow} />

      <TraceModal trace={selectedTrace} onClose={() => setSelectedTrace(null)} />
    </div>
  );
}
