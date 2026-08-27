const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* response wasn't JSON — keep statusText */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  runBatch: (n, useRealLlm) =>
    request("/batch/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n, use_real_llm: useRealLlm }),
    }),
  // Default policyName to "ai_agent": since the composite (transaction_id, policy_name)
  // ledger key, a batch_id can hold multiple policies (baseline experiment runs
  // do_nothing/generic_reminder under the same batch_id) — this dashboard shows the
  // live agent's own numbers by default, not a meaningless cross-policy sum. Pass
  // policyName={null} explicitly to opt out (mainly useful for debugging).
  metrics: (batchId, policyName = "ai_agent") => {
    const params = new URLSearchParams();
    if (batchId) params.set("batch_id", batchId);
    if (policyName) params.set("policy_name", policyName);
    const qs = params.toString();
    return request(`/metrics${qs ? `?${qs}` : ""}`);
  },
  batchDecisions: (batchId, policyName = "ai_agent") =>
    request(`/batch/${encodeURIComponent(batchId)}${policyName ? `?policy_name=${encodeURIComponent(policyName)}` : ""}`),
  compare: (batchId) => request(`/batch/${encodeURIComponent(batchId)}/compare`),
  audit: (transactionId) => request(`/audit/${encodeURIComponent(transactionId)}`),
};
