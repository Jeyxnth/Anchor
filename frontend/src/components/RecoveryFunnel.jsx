function formatINR(amount) {
  return `₹${amount.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

// Three-stage funnel replacing the cumulative-recovery-over-time line chart —
// no time-series here, just the headline shape of the batch: what was at
// risk, what came back, what's still exposed (at_risk - recovered).
export function RecoveryFunnel({ metrics }) {
  if (!metrics) {
    return <p className="muted">Run a batch to populate the funnel.</p>;
  }
  const atRisk = metrics.revenue_at_risk || 0;
  const recovered = metrics.revenue_recovered || 0;
  const stillAtRisk = Math.max(0, atRisk - recovered);

  return (
    <div className="funnel">
      <div className="funnel-block">
        <span className="funnel-label">₹ At Risk</span>
        <span className="funnel-value num">{formatINR(atRisk)}</span>
      </div>
      <div className="funnel-arrow" aria-hidden="true">→</div>
      <div className="funnel-block">
        <span className="funnel-label">₹ Recovered</span>
        <span className="funnel-value funnel-value--success num">{formatINR(recovered)}</span>
      </div>
      <div className="funnel-arrow" aria-hidden="true">→</div>
      <div className="funnel-block">
        <span className="funnel-label">₹ Still at Risk</span>
        <span className="funnel-value funnel-value--muted num">{formatINR(stillAtRisk)}</span>
      </div>
    </div>
  );
}
