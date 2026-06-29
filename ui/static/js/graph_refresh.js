/**
 * Graph Refresh Bridge
 * --------------------
 * This module avoids tight coupling between:
 * - main.js (where Cytoscape instance lives)
 * - other controllers (oversight / extracted_triplets)
 *
 * main.js registers a renderer callback once.
 * Any module can then call refreshGraphForKeyword(keyword) to refetch + re-render.
 */

import { getSubgraph, getFullGraph, getSearchKeywords } from "./graph_client.js";

let _renderSubgraph = null;

// Cache the curated dimension list so we can tell a real single-dimension
// keyword (valid for /subgraph) from a human run-label like "All dimensions",
// "Whole document", or "Custom: …" (which the subgraph endpoint rejects).
let _dimensionSet = null;
async function isCuratedDimension(keyword) {
    if (!keyword) return false;
    if (!_dimensionSet) {
        try {
            const data = await getSearchKeywords();
            _dimensionSet = new Set(data.keywords || []);
        } catch (_) {
            _dimensionSet = new Set();
        }
    }
    return _dimensionSet.has(keyword);
}

/**
 * Register a callback that knows how to render the subgraph payload.
 * Call this once from main.js after Cytoscape graph is created.
 *
 * @param {(payload: any) => void} fn
 */
export function registerSubgraphRenderer(fn) {
    _renderSubgraph = fn;
}

/**
 * Refetch and re-render the graph for a keyword.
 *
 * @param {string} keyword
 */
export async function refreshGraphForKeyword(keyword) {
    if (!_renderSubgraph) {
        throw new Error("Graph renderer not registered. Did you call registerSubgraphRenderer() in main.js?");
    }
    // Single curated dimensions can be fetched as a focused subgraph. Multi-
    // dimension ("All dimensions"), whole-document, and custom runs carry a
    // human label that isn't a valid subgraph keyword — show the full graph.
    const payload = (await isCuratedDimension(keyword))
        ? await getSubgraph(keyword)
        : await getFullGraph();
    _renderSubgraph(payload);
}
