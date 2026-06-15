/**
 * API client for the Compare tab (/api/comparison/*).
 *
 * Background scans (alignment, conflicts) return a job_id polled via the
 * shared pipeline jobs endpoint.
 */

import { fetchJson } from "./api_client.js";

export async function fetchOverlapReport() {
  return fetchJson("/api/comparison/overlap");
}

export async function startAlignmentScan({ threshold, max_candidates } = {}) {
  return fetchJson("/api/comparison/alignment/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threshold, max_candidates }),
  });
}

export async function postAlignmentDecision({ term_a, term_b, accept, score }) {
  return fetchJson("/api/comparison/alignment/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ term_a, term_b, accept, score }),
  });
}

export async function startConflictScan({ judge = true } = {}) {
  return fetchJson("/api/comparison/conflicts/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ judge }),
  });
}

export async function postConflictDecision({ candidate, accept }) {
  return fetchJson("/api/comparison/conflicts/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate, accept }),
  });
}

export async function fetchRecordedConflicts() {
  return fetchJson("/api/comparison/conflicts/recorded");
}

export async function fetchAmbiguityReport() {
  return fetchJson("/api/comparison/ambiguity");
}
