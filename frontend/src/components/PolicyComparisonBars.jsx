const POLICY_LABELS = {
  ai_agent: "AI Agent",
  generic_reminder: "Generic Reminder",
  do_nothing: "Do Nothing",
};

function formatINR(amount) {
  return `₹${amount.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

// Condensed headline version of the full Policy Comparison section further
// down the page — same `comparison` data (from the same baseline experiment
// run/load there), just three bars sorted ascending by ₹ recovered so the
// payoff reads at a glance from the top of the dashboard.
export function PolicyComparisonBars({ comparison }) {
  const results = comparison?.results;
  if (!results) {
    return <p className="muted">Run the baseline experiment below to populate this.</p>;
  }

  const rows = Object.entries(results)
    .map(([policy, r]) => ({ policy, recovered: r.revenue_recovered || 0 }))
    .sort((a, b) => a.recovered - b.recovered);
  const max = Math.max(1, ...rows.map((r) => r.recovered));

  return (
    <div className="policy-bars">
      {rows.map((r) => {
        const isAgent = r.policy === "ai_agent";
        return (
          <div className="policy-bar-row" key={r.policy}>
            <div className="policy-bar-head">
              <span className={`policy-bar-label${isAgent ? " policy-bar-label--agent" : ""}`}>
                {POLICY_LABELS[r.policy] || r.policy}
              </span>
              <span className="policy-bar-value num">{formatINR(r.recovered)}</span>
            </div>
            <div className="policy-bar-track">
              <div
                className={`policy-bar-fill ${isAgent ? "policy-bar-fill--accent" : "policy-bar-fill--muted"}`}
                style={{ width: `${(r.recovered / max) * 100}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
