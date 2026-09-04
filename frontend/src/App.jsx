import { useEffect, useState } from "react";
import {
  IconCheck,
  IconCurrencyRupee,
  IconInfoCircle,
  IconLayoutDashboard,
  IconListDetails,
  IconPercentage,
  IconScale,
  IconSearch,
  IconShieldCheck,
  IconTrendingUp,
} from "@tabler/icons-react";
import {
  Anchor,
  ArrowRight,
  Bell,
  Brain,
  Check,
  CheckCircle2,
  Dices,
  Filter as FilterIcon,
  ListOrdered,
  RefreshCw,
  ShieldCheck,
  Tag,
  User,
  X as XLucide,
  XCircle,
} from "lucide-react";
import { api } from "./api";
import { RecoveryFunnel } from "./components/RecoveryFunnel";
import { PolicyComparisonBars } from "./components/PolicyComparisonBars";
import "./App.css";

const ACTION_LABELS = {
  retry_link: "Retry Link",
  reminder: "Reminder",
  discount_offer: "Discount Offer",
  escalate_to_human: "Escalate to Human",
  no_action: "No Action",
};

// Per-action icons, used everywhere an action appears (trace modal, Recent
// Decisions row avatars). lucide-react per the redesign spec.
const ACTION_ICONS = {
  retry_link: RefreshCw,
  reminder: Bell,
  discount_offer: Tag,
  escalate_to_human: User,
  no_action: XLucide,
};

const ACTION_COLORS = {
  retry_link: "var(--chart-1)",
  reminder: "var(--chart-2)",
  discount_offer: "var(--chart-3)",
  escalate_to_human: "var(--chart-4)",
  no_action: "var(--chart-5)",
};

// Five-stage pipeline shown at the top of every trace modal — every trace
// audited here already ran end to end, so all five always render "done".
const TRACE_STAGES = [
  { id: "predict", label: "Predict", icon: Brain },
  { id: "filter", label: "Filter", icon: FilterIcon },
  { id: "rank", label: "Rank", icon: ListOrdered },
  { id: "decide", label: "Decide", icon: Check },
  { id: "simulate", label: "Simulate", icon: Dices },
];

const POLICY_LABELS = {
  ai_agent: "AI Agent",
  generic_reminder: "Generic Reminder",
  do_nothing: "Do Nothing",
};
const POLICY_ORDER = ["do_nothing", "generic_reminder", "ai_agent"];

const DEMO_CASES = [
  { id: "TXN000000", label: "Normal case", hint: "all 5 actions eligible" },
  { id: "TXN000025", label: "Opted-out", hint: "compliance restricts to no_action only" },
  { id: "TXN002166", label: "Contact-capped", hint: "compliance restricts to escalate/no_action" },
];

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: IconLayoutDashboard },
  { id: "policy-comparison", label: "Policy Comparison", icon: IconScale },
  { id: "compliance-demo-cases", label: "Compliance Demo Cases", icon: IconShieldCheck },
  { id: "recent-decisions", label: "Recent Decisions", icon: IconListDetails },
];

function formatINR(amount) {
  if (amount === null || amount === undefined) return "—";
  return `₹${amount.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

function formatPct(fraction) {
  if (fraction === null || fraction === undefined) return "—";
  return `${(fraction * 100).toFixed(1)}%`;
}

function formatDate(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

function PendingNote({ children }) {
  return <span className="pending-note" title={children}>pending</span>;
}

function SectionHead({ children, subtitle, info }) {
  return (
    <div className="section-head">
      <h2 className="section-title">
        {children}
        {info && <InfoTip text={info} />}
      </h2>
      {subtitle && <p className="section-subtitle">{subtitle}</p>}
    </div>
  );
}

// Small "i" badge that reveals detail on hover/focus — used to move
// permanent multi-sentence footnotes off the page and into an on-demand
// tooltip instead.
function InfoTip({ text }) {
  return (
    <span className="info-tip" tabIndex={0}>
      <IconInfoCircle size={14} />
      <span className="info-tip-bubble">{text}</span>
    </span>
  );
}

/* ---------------------------------------------------------------------- */

function Sidebar({ apiUp }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-mark">
          <Anchor size={17} strokeWidth={2.25} />
        </span>
        <span className="sidebar-brand-name">ANCHOR</span>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <a href={`#${item.id}`} key={item.id} className="sidebar-nav-item">
              <Icon size={17} />
              <span>{item.label}</span>
            </a>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <span className={`sidebar-status-dot ${apiUp ? "sidebar-status-dot--up" : "sidebar-status-dot--down"}`} />
        <span>{apiUp === null ? "checking API…" : apiUp ? "API connected" : "API unreachable"}</span>
      </div>
    </aside>
  );
}

function HeaderSearch({ value, onChange, onSubmit }) {
  return (
    <div className="header-search">
      <IconSearch size={16} className="header-search-icon" />
      <input
        type="text"
        placeholder="Jump to transaction (e.g. TXN000025)"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && value && onSubmit(value.trim())}
      />
    </div>
  );
}

// Headline figures target 48-64px, but a long rupee string at 64px will
// wrap inside a ~260px card — so step down for longer strings rather than
// let it wrap (previous bug: see git history).
function kpiFontSize(value) {
  const len = String(value ?? "").length;
  if (len <= 7) return "4rem"; // 64px
  if (len <= 9) return "3.4rem"; // ~54px
  if (len <= 11) return "3rem"; // 48px
  if (len <= 13) return "2.35rem";
  return "1.9rem";
}

function KpiCard({ icon: Icon, iconTone, label, value, delta, deltaTone }) {
  return (
    <div className="kpi-card">
      <div className="kpi-card-top">
        <span className="kpi-card-label">{label}</span>
        <span className={`kpi-card-icon kpi-card-icon--${iconTone}`}>
          <Icon size={16} />
        </span>
      </div>
      <div className="kpi-card-value num" style={{ fontSize: kpiFontSize(value) }}>
        {value === null ? "—" : value}
      </div>
      {delta && <div className={`kpi-card-delta ${deltaTone ? `kpi-card-delta--${deltaTone}` : ""}`}>{delta}</div>}
    </div>
  );
}

function HeadlineKpis({ metrics, comparison }) {
  const incrementalRaw = comparison?.incremental_recovery_of_ai_agent_vs_each_baseline
    ? Math.min(...Object.values(comparison.incremental_recovery_of_ai_agent_vs_each_baseline))
    : null;

  return (
    <div className="kpi-row">
      <KpiCard
        icon={IconCurrencyRupee}
        iconTone="accent"
        label="Revenue at Risk"
        value={metrics ? formatINR(metrics.revenue_at_risk) : null}
        delta={metrics ? "across current batch" : "run a batch to populate"}
      />
      <KpiCard
        icon={IconCheck}
        iconTone="success"
        label="Revenue Recovered"
        value={metrics && metrics.revenue_recovered !== null ? formatINR(metrics.revenue_recovered) : null}
        delta={metrics?.revenue_weighted_recovery_rate !== null ? `${formatPct(metrics?.revenue_weighted_recovery_rate)} of ₹ at risk` : null}
        deltaTone="success"
      />
      <KpiCard
        icon={IconPercentage}
        iconTone="accent"
        label="Recovery Rate"
        value={metrics && metrics.recovery_rate !== null ? formatPct(metrics.recovery_rate) : null}
        delta="count-based: recovered / total txns"
      />
      <KpiCard
        icon={IconTrendingUp}
        iconTone={incrementalRaw !== null && incrementalRaw > 0 ? "success" : "muted"}
        label="Lift vs. Best Baseline"
        value={incrementalRaw !== null ? formatINR(incrementalRaw) : null}
        delta={comparison ? "vs. generic_reminder (stronger baseline)" : "run baseline experiment below"}
        deltaTone={incrementalRaw !== null && incrementalRaw > 0 ? "success" : incrementalRaw !== null ? "danger" : null}
      />
    </div>
  );
}

function StoppingReasons({ breakdown }) {
  const entries = Object.entries(breakdown || {});
  if (entries.length === 0) return null;
  return (
    <div className="stopping-reasons">
      <span className="stopping-reasons-label">Stopping reasons:</span>
      {entries.map(([reason, n]) => (
        <span key={reason} className="stopping-reasons-item">
          <span className="num">{n}</span> {reason.replace(/_/g, " ")}
        </span>
      ))}
    </div>
  );
}

function CompliancePanel({ compliance, nCases, stoppingReasonBreakdown }) {
  if (!compliance) return null;
  const capBar = { value: compliance.contact_cap_restricted, max: Math.max(1, nCases) };
  // opted_out_respected is always 100% by construction (the eligibility
  // filter never fails to restrict an opted-out case) — the bar is either
  // empty (no opted-out customers this batch) or full, never a real
  // fraction, so "N of N" reads honestly instead of implying a synthetic
  // denominator like "0 / 1" would.

  return (
    <section className="card" id="compliance-panel">
      <SectionHead
        info={'Only "recovered" / "not_recovered" outcomes exist today — this scores one event per transaction, not a multi-touch retry sequence, so richer stopping reasons aren\'t possible yet.'}
      >
        Compliance Panel
      </SectionHead>
      <div className="progress-list">
        <div className="progress-row">
          <div className="progress-row-head">
            <span>Opt-outs respected</span>
            <span className="num">
              {compliance.opted_out_respected === 0
                ? "0 opted-out customers this batch"
                : `${compliance.opted_out_respected} of ${compliance.opted_out_respected} respected`}
            </span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill progress-fill--success"
              style={{ width: compliance.opted_out_respected === 0 ? "0%" : "100%" }}
            />
          </div>
        </div>
        <div className="progress-row">
          <div className="progress-row-head">
            <span>Contact-limit restricted</span>
            <span className="num">{compliance.contact_cap_restricted} / {capBar.max}</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill progress-fill--warning" style={{ width: `${(capBar.value / capBar.max) * 100}%` }} />
          </div>
        </div>
        <div className="progress-row">
          <div className="progress-row-head">
            <span>Quiet-hour violations</span>
            <span>{compliance.quiet_hour_violations === null ? <PendingNote>{compliance._note}</PendingNote> : compliance.quiet_hour_violations}</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill progress-fill--muted" style={{ width: "0%" }} />
          </div>
        </div>
      </div>
      <div className="compliance-banner">
        ✓ {compliance.target_compliance_violations} target compliance violations across {nCases} cases — on track.
      </div>
      <StoppingReasons breakdown={stoppingReasonBreakdown} />
    </section>
  );
}

function PolicyComparisonBlock({ comparison, loading, batchIdInput, onBatchIdInputChange, onLoad, onRunBaseline, running }) {
  const results = comparison?.results;
  const maxRecovered = results
    ? Math.max(1, ...POLICY_ORDER.map((p) => results[p]?.revenue_recovered || 0))
    : 1;

  return (
    <section className="card" id="policy-comparison">
      <div className="policy-head">
        <SectionHead
          subtitle="Same transactions, three policies, real simulated outcomes."
          info="Compares do_nothing, generic_reminder, and ai_agent by replaying the same transaction population under each policy so the result is apples-to-apples."
        >
          Policy Comparison — Baseline Experiment
        </SectionHead>
        <div className="policy-head-controls">
          <input
            type="text"
            className="field-input"
            placeholder="batch id to load"
            value={batchIdInput}
            onChange={(e) => onBatchIdInputChange(e.target.value)}
          />
          <button className="btn btn--ghost" onClick={onLoad} disabled={loading || !batchIdInput}>Load</button>
          <button className="btn btn--primary" onClick={onRunBaseline} disabled={running}>
            {running ? "Running…" : "Run Baseline Experiment"}
          </button>
        </div>
      </div>

      {!comparison ? (
        <p className="muted">Load a batch id or run a fresh baseline experiment to compare.</p>
      ) : (
        <>
          <p className="muted" style={{ marginBottom: 20 }}>
            {comparison.n_cases_per_policy?.ai_agent ?? "—"} transactions · {formatINR(comparison.revenue_at_risk)} at risk
            {comparison.same_transaction_population_across_policies
              ? " · same population across all three policies (fair comparison)"
              : " · ⚠ populations differ across policies — comparison is NOT apples-to-apples"}
          </p>

          <div className="policy-grid">
            {POLICY_ORDER.map((p) => {
              const r = results[p];
              if (!r) return null;
              const isAgent = p === "ai_agent";
              return (
                <div key={p} className={isAgent ? "policy-card policy-card--agent" : "policy-card"}>
                  <div className="policy-card-name">{POLICY_LABELS[p] || p}</div>
                  <div className="policy-card-value num">{formatINR(r.revenue_recovered)}</div>
                  <div className="policy-card-rate">
                    {formatPct(r.recovery_rate)} recovered
                    <span className="policy-card-rate-secondary"> ({formatPct(r.revenue_weighted_recovery_rate)} of ₹ at risk)</span>
                  </div>
                  <div className="progress-track" style={{ marginTop: 10 }}>
                    <div
                      className={`progress-fill ${isAgent ? "progress-fill--accent" : "progress-fill--muted"}`}
                      style={{ width: `${(r.revenue_recovered / maxRecovered) * 100}%` }}
                    />
                  </div>
                  {r.incremental_recovery_vs_ai_agent !== null && (
                    <div className={`policy-card-incremental ${r.incremental_recovery_vs_ai_agent >= 0 ? "text-success" : "text-danger"}`}>
                      {r.incremental_recovery_vs_ai_agent >= 0
                        ? `agent recovers +${formatINR(r.incremental_recovery_vs_ai_agent)} more`
                        : `agent recovers ${formatINR(Math.abs(r.incremental_recovery_vs_ai_agent))} less`}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <p className="muted footnote-with-tip" style={{ marginTop: 20 }}>
            Recovery rate is count-based; the parenthetical is ₹-weighted.
            <InfoTip text="Count-based = recovered transactions / total transactions. ₹-weighted = ₹ recovered / ₹ at risk. The two won't match numerically, by design." />
          </p>
        </>
      )}
    </section>
  );
}

function DemoLinks({ onOpen }) {
  return (
    <section className="card" id="compliance-demo-cases">
      <SectionHead
        subtitle="Click-through cases for the compliance demo."
        info="Two compliance-restricted cases plus one normal case for contrast — no need to hunt for transaction IDs. Full walkthrough in DEMO_SCRIPT.md."
      >
        Compliance Demo Cases
      </SectionHead>
      <div className="demo-grid">
        {DEMO_CASES.map((c) => (
          <button key={c.id} onClick={() => onOpen(c.id)} className="demo-card">
            <span className="demo-card-id num">{c.id}</span>
            <span className="demo-card-label">{c.label}</span>
            <span className="demo-card-hint">{c.hint}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function DecisionsList({ decisions, onSelect }) {
  return (
    <section className="card" id="recent-decisions">
      <div className="section-head-row">
        <SectionHead>Recent Decisions</SectionHead>
        {decisions.length > 0 && <span className="muted">{decisions.length} shown</span>}
      </div>
      {decisions.length === 0 ? (
        <p className="muted">No decisions logged yet — run a batch above.</p>
      ) : (
        <div className="activity-list">
          <div className="activity-row activity-row--header" aria-hidden="true">
            <span />
            <span className="activity-col-label">Action</span>
            <span className="activity-col-label activity-col-label--right">Date</span>
            <span className="activity-col-label activity-col-label--right">Amount</span>
            <span className="activity-col-label">Status</span>
            <span className="activity-col-label">Provider</span>
          </div>
          {decisions.map((d) => {
            const ActionIcon = ACTION_ICONS[d.selected_action] || XLucide;
            return (
              <div key={d.transaction_id} className="activity-row" onClick={() => onSelect(d.transaction_id)}>
                <span className="activity-avatar" style={{ background: `color-mix(in srgb, ${ACTION_COLORS[d.selected_action] || "var(--chart-5)"} 16%, white)`, color: ACTION_COLORS[d.selected_action] || "var(--chart-5)" }}>
                  <ActionIcon size={16} />
                </span>
                <div className="activity-main">
                  <div className="activity-title">{ACTION_LABELS[d.selected_action] || d.selected_action}</div>
                  <div className="activity-subtitle num">{d.transaction_id} · {d.failure_reason}</div>
                </div>
                <div className="activity-date">{formatDate(d.created_at)}</div>
                <div className={`activity-amount num ${d.outcome_recovered ? "text-success" : "text-muted"}`}>
                  {d.outcome_recovered ? `+${formatINR(d.outcome_recovered_amount)}` : "₹0"}
                </div>
                <div className="activity-col-badge">
                  <span className={`pill ${d.outcome_recovered ? "pill--success" : "pill--danger"}`}>
                    {d.outcome_recovered ? "Recovered" : "Not Recovered"}
                  </span>
                </div>
                <div className="activity-col-badge">
                  <span className={`pill ${d.agent_provider === "rule_based_fallback" ? "pill--warning" : "pill--accent"}`}>
                    {d.agent_provider}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function StageTracker() {
  return (
    <div className="stage-tracker">
      {TRACE_STAGES.map((s) => {
        const Icon = s.icon;
        return (
          <div className="stage-tracker-item" key={s.id}>
            <span className="stage-node stage-node--done">
              <Icon size={15} />
            </span>
            <span className="stage-label">{s.label}</span>
          </div>
        );
      })}
    </div>
  );
}

// Shared row for both the probability list and the EV-ranked list — icon,
// name, a horizontal bar scaled to `pct`, and the formatted value inline at
// the bar's end. Ineligible/filtered-out actions render as muted grey;
// the winning action gets a highlighted row background.
function ActionBarRow({ action, pct, displayValue, eligible, highlighted, meta }) {
  const Icon = ACTION_ICONS[action] || XLucide;
  return (
    <div className={`action-bar-row${highlighted ? " action-bar-row--selected" : ""}`}>
      <span className={`action-bar-icon${eligible ? "" : " action-bar-icon--muted"}`}>
        <Icon size={15} />
      </span>
      <span className="action-bar-name">{ACTION_LABELS[action] || action}</span>
      <div className="action-bar-track">
        <div
          className={`action-bar-fill ${eligible ? "action-bar-fill--accent" : "action-bar-fill--muted"}`}
          style={{ width: `${Math.max(2, pct)}%` }}
        />
      </div>
      <span className="action-bar-value num">{displayValue}</span>
      {meta && <span className="action-bar-meta">{meta}</span>}
    </div>
  );
}

function PredictedProbabilityBars({ predicted, eligibleActions, selectedAction }) {
  const entries = Object.entries(predicted);
  if (entries.length === 0) {
    return <p className="muted">Not applicable — this policy doesn't consult the model (fixed-rule baseline).</p>;
  }
  return (
    <div className="action-bar-list">
      {entries.map(([a, p]) => (
        <ActionBarRow
          key={a}
          action={a}
          pct={p * 100}
          displayValue={formatPct(p)}
          eligible={eligibleActions.includes(a)}
          highlighted={a === selectedAction}
        />
      ))}
    </div>
  );
}

function EvRankedBars({ candidates, selectedAction }) {
  if (candidates.length === 0) return null;
  const maxEv = Math.max(...candidates.map((c) => c.expected_value), 0.01);
  return (
    <div className="action-bar-list">
      {candidates.map((c) => (
        <ActionBarRow
          key={c.action}
          action={c.action}
          pct={(c.expected_value / maxEv) * 100}
          displayValue={formatINR(c.expected_value)}
          eligible
          highlighted={c.action === selectedAction}
          meta={`cost ${formatINR(c.intervention_cost)} · discount ${formatINR(c.discount_cost)}`}
        />
      ))}
    </div>
  );
}

function EligibleActionChips({ actions }) {
  return (
    <div className="eligible-chip-row">
      {actions.map((a) => {
        const Icon = ACTION_ICONS[a] || XLucide;
        return (
          <span className="eligible-chip" key={a}>
            <Icon size={13} />
            {ACTION_LABELS[a] || a}
          </span>
        );
      })}
    </div>
  );
}

function ComplianceEffectCards({ trace }) {
  const changed = trace.compliance_changed_top_action;
  const unconstrainedTop = trace.candidates_ev_ranked_unconstrained[0];
  const UnIcon = ACTION_ICONS[trace.unconstrained_top_action] || XLucide;
  const SelIcon = ACTION_ICONS[trace.selected_action] || XLucide;
  const selectedEv = trace.candidates_ev_ranked.find((c) => c.action === trace.selected_action);

  return (
    <div>
      <div className="effect-cards">
        <div className="effect-card">
          <span className="effect-card-label">Unconstrained top pick</span>
          <div className="effect-card-action">
            <span className="effect-card-icon">
              <UnIcon size={20} />
            </span>
            <span className="effect-card-name">{ACTION_LABELS[trace.unconstrained_top_action] || trace.unconstrained_top_action}</span>
          </div>
          <span className="effect-card-ev num">EV {formatINR(unconstrainedTop.expected_value)}</span>
        </div>
        <div className={`effect-arrow ${changed ? "effect-arrow--changed" : "effect-arrow--same"}`}>
          {changed ? <ArrowRight size={18} /> : <Check size={18} />}
        </div>
        <div className={`effect-card effect-card--actual${changed ? " effect-card--changed" : ""}`}>
          <span className="effect-card-label">Actual decision</span>
          <div className="effect-card-action">
            <span className="effect-card-icon">
              <SelIcon size={20} />
            </span>
            <span className="effect-card-name">{ACTION_LABELS[trace.selected_action] || trace.selected_action}</span>
          </div>
          <span className="effect-card-ev num">EV {selectedEv ? formatINR(selectedEv.expected_value) : "—"}</span>
        </div>
      </div>
      <p className="muted" style={{ marginTop: 12 }}>
        {changed
          ? `Compliance restricted the eligible set to ${trace.eligible_actions.map((a) => ACTION_LABELS[a] || a).join(", ")}, changing the outcome.`
          : "Unconstrained pick was already eligible — compliance didn't change the outcome."}
      </p>
    </div>
  );
}

function DecisionCard({ trace }) {
  const Icon = ACTION_ICONS[trace.selected_action] || XLucide;
  return (
    <div className="status-card">
      <span className="status-card-icon status-card-icon--accent">
        <Icon size={24} />
      </span>
      <div className="status-card-body">
        <div className="status-card-title-row">
          <span className="status-card-title">{ACTION_LABELS[trace.selected_action] || trace.selected_action}</span>
          <span className={`pill ${trace.agent_provider === "rule_based_fallback" ? "pill--warning" : "pill--accent"}`}>
            {trace.agent_provider}
          </span>
        </div>
        <p className="status-card-detail">{trace.agent_reason}</p>
        {!trace.agent_valid && <p className="warning-note">⚠ {trace.agent_validation_note}</p>}
      </div>
    </div>
  );
}

function OutcomeCard({ trace }) {
  const recovered = trace.outcome_recovered;
  return (
    <div className="status-card">
      <span className={`status-card-icon ${recovered ? "status-card-icon--success" : "status-card-icon--danger"}`}>
        {recovered ? <CheckCircle2 size={24} /> : <XCircle size={24} />}
      </span>
      <div className="status-card-body">
        <span className="status-card-title">
          {recovered ? `Recovered ${formatINR(trace.outcome_recovered_amount)}` : "Not recovered"}
        </span>
        <p className="status-card-detail">
          {recovered ? `In ${trace.time_to_recovery_hours}h · ` : ""}Simulated against ground_truth.csv's true probability.
        </p>
      </div>
    </div>
  );
}

function TraceModal({ trace, onClose }) {
  if (!trace) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="num">{trace.transaction_id} — full decision trace</h3>
          <button onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          <StageTracker />

          <section>
            <h4>Context</h4>
            <p>
              {trace.customer_id} · <span className="num">{formatINR(trace.amount)}</span> · {trace.event_type} / {trace.failure_reason} ·
              policy: <strong>{POLICY_LABELS[trace.policy_name] || trace.policy_name}</strong>
            </p>
          </section>

          <section>
            <h4>Predicted recovery probability per action</h4>
            <PredictedProbabilityBars
              predicted={trace.predicted_probabilities}
              eligibleActions={trace.eligible_actions}
              selectedAction={trace.selected_action}
            />
          </section>

          <section>
            <h4>Eligible actions (compliance filter)</h4>
            <EligibleActionChips actions={trace.eligible_actions} />
          </section>

          {trace.candidates_ev_ranked_unconstrained?.length > 0 && (
            <section>
              <h4>Compliance effect — before / after</h4>
              <ComplianceEffectCards trace={trace} />
            </section>
          )}

          {trace.candidates_ev_ranked.length > 0 && (
            <section>
              <h4>EV-ranked candidates (eligible only)</h4>
              <EvRankedBars candidates={trace.candidates_ev_ranked} selectedAction={trace.selected_action} />
            </section>
          )}

          <section className="status-card-grid">
            <DecisionCard trace={trace} />
            <OutcomeCard trace={trace} />
          </section>

          <section className="compliance-callout">
            <span className="compliance-callout-icon">
              <ShieldCheck size={18} />
            </span>
            <p>
              Compliance is enforced two ways: eligible actions are filtered before the agent ever runs, and
              every decision is independently re-validated against that same list. See the Compliance Panel on
              the dashboard for this batch's stats.
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
  const [runningBaseline, setRunningBaseline] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [apiUp, setApiUp] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [compareBatchIdInput, setCompareBatchIdInput] = useState("");
  const [jumpId, setJumpId] = useState("");

  useEffect(() => {
    api.health().then(() => setApiUp(true)).catch(() => setApiUp(false));
  }, []);

  async function refresh(id) {
    const [m, b] = await Promise.all([api.metrics(id), api.batchDecisions(id)]);
    setMetrics(m);
    setDecisions(b.decisions);
  }

  async function loadComparison(id) {
    setComparisonLoading(true);
    setError(null);
    try {
      const c = await api.compare(id);
      setComparison(c);
      setCompareBatchIdInput(id);
    } catch (e) {
      setError(e.message);
    } finally {
      setComparisonLoading(false);
    }
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

  async function handleRunBaselineExperiment() {
    setRunningBaseline(true);
    setError(null);
    try {
      const result = await api.runBaselineExperiment(nRows || null, useRealLlm, null);
      setBatchId(result.batch_id);
      setComparison(result.comparison);
      setCompareBatchIdInput(result.batch_id);
      await refresh(result.batch_id);
    } catch (e) {
      setError(e.message);
    } finally {
      setRunningBaseline(false);
    }
  }

  async function handleOpenTrace(transactionId) {
    try {
      const trace = await api.audit(transactionId);
      setSelectedTrace(trace);
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar apiUp={apiUp} />

      <main className="content">
        <header className="content-header" id="dashboard">
          <div className="content-header-title">
            <h1 className="page-title">Revenue Recovery Control Center</h1>
            <p className="page-subtitle">Real-time batch decisions, compliance, and recovered revenue.</p>
            {batchId && <p className="page-batch-note">Active batch: <span className="num">{batchId}</span></p>}
          </div>
          <HeaderSearch value={jumpId} onChange={setJumpId} onSubmit={handleOpenTrace} />
          <div className="header-controls">
            <div className="field">
              <label htmlFor="rows-input" className="field-label">rows</label>
              <input
                id="rows-input"
                type="number"
                min="1"
                max="4000"
                className="field-input field-input--narrow"
                value={nRows}
                onChange={(e) => setNRows(Number(e.target.value))}
              />
            </div>
            <label className="toggle">
              <input type="checkbox" checked={useRealLlm} onChange={(e) => setUseRealLlm(e.target.checked)} />
              <span className="toggle-track"><span className="toggle-thumb" /></span>
              use real LLM
            </label>
            <button className="btn btn--primary" onClick={handleRunBatch} disabled={running}>
              {running ? "Running…" : "Run Batch"}
            </button>
          </div>
        </header>

        {error && <div className="error-banner">Error: {error}</div>}

        <HeadlineKpis metrics={metrics} comparison={comparison} />

        <div className="two-col">
          <section className="card">
            <SectionHead subtitle="₹ at risk, recovered, and still exposed.">Recovery Funnel</SectionHead>
            <RecoveryFunnel metrics={metrics} />
          </section>
          <section className="card">
            <SectionHead subtitle="₹ recovered by policy, same population.">Policy Comparison</SectionHead>
            <PolicyComparisonBars comparison={comparison} />
          </section>
        </div>

        <CompliancePanel
          compliance={metrics?.compliance}
          nCases={metrics?.n_cases || 0}
          stoppingReasonBreakdown={metrics?.stopping_reason_breakdown}
        />

        <PolicyComparisonBlock
          comparison={comparison}
          loading={comparisonLoading}
          batchIdInput={compareBatchIdInput}
          onBatchIdInputChange={setCompareBatchIdInput}
          onLoad={() => loadComparison(compareBatchIdInput)}
          onRunBaseline={handleRunBaselineExperiment}
          running={runningBaseline}
        />

        <DemoLinks onOpen={handleOpenTrace} />

        <DecisionsList decisions={decisions} onSelect={handleOpenTrace} />
      </main>

      <TraceModal trace={selectedTrace} onClose={() => setSelectedTrace(null)} />
    </div>
  );
}
