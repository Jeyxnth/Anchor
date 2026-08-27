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
  metrics: (batchId) => request(`/metrics${batchId ? `?batch_id=${encodeURIComponent(batchId)}` : ""}`),
  batchDecisions: (batchId) => request(`/batch/${encodeURIComponent(batchId)}`),
  audit: (transactionId) => request(`/audit/${encodeURIComponent(transactionId)}`),
};
