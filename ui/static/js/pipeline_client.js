import { fetchJson } from "./api_client.js";

export async function startPipelineJob({ keyword, files }) {
  const form = new FormData();
  for (const file of files || []) {
    form.append("documents", file);
  }

  const url = `/api/pipeline/run?keyword=${encodeURIComponent(keyword)}`;
  return fetchJson(url, { method: "POST", body: form });
}

export async function getJobStatus(jobId) {
  return fetchJson(`/api/pipeline/jobs/${encodeURIComponent(jobId)}`);
}


export async function startTripletExtractionJob({ selected_items }) {
  return fetchJson("/api/pipeline/triplet-extraction", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ selected_items }),
  });
}

export async function upsertTripletsToKnowledgeGraphJob({ extractions, source }) {
  return fetchJson("/api/pipeline/kg-upsert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ extractions, source }),
  });
}
