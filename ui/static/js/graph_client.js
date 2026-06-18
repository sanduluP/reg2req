import { fetchJson } from "./api_client.js";

export async function getSearchKeywords(options = {}) {
  return fetchJson("/api/graph/search-keywords", options);
}

export async function getSubgraph(keyword, options = {}) {
  const url = `/api/graph/subgraph?keyword=${encodeURIComponent(keyword)}`;
  return fetchJson(url, options);
}

export async function getFullGraph(options = {}) {
  return fetchJson("/api/graph/full", options);
}

export async function seedGraph({ reset = true } = {}) {
  return fetchJson("/api/graph/seed", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reset }),
  });
}
