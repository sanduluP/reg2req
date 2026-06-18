/**
 * API client for the Compare tab (/api/comparison/*).
 *
 * Background scans (alignment, conflicts) return a job_id polled via the
 * shared pipeline jobs endpoint.
 *
 * All analysis calls accept an optional `sources` array that restricts the
 * provenance-layer analysis to the selected document sources.
 */

import { fetchJson } from "./api_client.js";

/** Return unique provenance_source values from Neo4j. */
export async function fetchComparisonSources() {
  return fetchJson("/api/comparison/sources");
}

/**
 * @param {string[]|null} [sources]  null = all sources
 */
export async function fetchOverlapReport({ sources } = {}) {
  const params = _sourcesParam(sources);
  const url = params ? `/api/comparison/overlap?${params}` : "/api/comparison/overlap";
  return fetchJson(url);
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

/**
 * @param {boolean} [judge]
 * @param {string[]|null} [sources]
 */
export async function startConflictScan({ judge = true, sources } = {}) {
  return fetchJson("/api/comparison/conflicts/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ judge, sources: sources ?? null }),
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

/**
 * @param {string[]|null} [sources]
 */
export async function fetchAmbiguityReport({ sources } = {}) {
  const params = _sourcesParam(sources);
  const url = params ? `/api/comparison/ambiguity?${params}` : "/api/comparison/ambiguity";
  return fetchJson(url);
}

/** Build a `sources=a,b,c` query param string, or null if no filter. */
function _sourcesParam(sources) {
  if (!Array.isArray(sources) || sources.length === 0) return null;
  return `sources=${encodeURIComponent(sources.join(","))}`;
}
