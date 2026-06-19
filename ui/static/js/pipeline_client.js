import { fetchJson } from "./api_client.js";

export async function startPipelineJob({ keyword, custom_keywords, files, thresholds }) {
  const form = new FormData();
  for (const file of files || []) {
    form.append("documents", file);
  }

  // Custom keyword(s) for the "Custom keyword(s)…" mode (comma-separated).
  if (custom_keywords) form.append("custom_keywords", custom_keywords);

  // Per-run threshold overrides (optional). Only non-empty values are sent;
  // the backend falls back to env defaults for anything omitted.
  for (const key of ["para_threshold", "sim_threshold", "top_k", "kg_limit"]) {
    const v = thresholds?.[key];
    if (v !== undefined && v !== null && v !== "") form.append(key, String(v));
  }

  const url = `/api/pipeline/run?keyword=${encodeURIComponent(keyword)}`;
  return fetchJson(url, { method: "POST", body: form });
}

export async function getPipelineThresholds() {
  return fetchJson("/api/pipeline/thresholds");
}

export async function getJobStatus(jobId) {
  return fetchJson(`/api/pipeline/jobs/${encodeURIComponent(jobId)}`);
}

/**
 * Incrementally extend extraction to chunks newly included by a lower relevance
 * threshold. Returns { job_id } to poll; the result carries the new results.
 * The HTTP 404 (context expired) is surfaced so the caller can fall back to the
 * manual re-run hint.
 */
export async function startExtendExtractionJob({ source_job_id, threshold }) {
  return fetchJson("/api/pipeline/extend-extraction", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_job_id, threshold }),
  });
}


export async function startTripletExtractionJob({ selected_items, extraction_settings }) {
  return fetchJson("/api/pipeline/triplet-extraction", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ selected_items, extraction_settings }),
  });
}

export async function upsertTripletsToKnowledgeGraphJob({ extractions, source }) {
  return fetchJson("/api/pipeline/kg-upsert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ extractions, source }),
  });
}
