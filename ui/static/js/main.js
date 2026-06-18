/**
 * Main entrypoint:
 * - initialize graph controller (Cytoscape)
 * - initialize keyword dropdown and connect it to graph updates
 *
 * Keep this file tiny.
 */

import { createCytoscapeGraph } from "./cytoscape.js";
import { initKeywordDropdown } from "./keyword_dropdown_controller.js";
import { wirePipelineRunControls } from "./pipeline_controller.js";
import { wireHumanOversightSubmit, renderHumanOversightFromPipelineResult, wireGoToTripletsButton } from "./oversight_controller.js";
import { wireComparisonView, refreshComparisonSources } from "./comparison_controller.js";
import { wireSeedButton } from "./seed_controller.js";
import { registerSubgraphRenderer } from "./graph_refresh.js";
import { hideProgressPanel } from "./pipeline_progress_ui.js"

// Optional modules (TODO: Most likely will be deleted)
// import { wireGraphSearch } from "./graph_free_text_search.js";

document.addEventListener("DOMContentLoaded", async () => {
  // ---------------------------------------------------------------------------
  // 1) Graph initialize (Cytoscape)
  // ---------------------------------------------------------------------------
  const graph = createCytoscapeGraph("cy", "details-content");

  // To make our graph refreshable from anywhere.
  registerSubgraphRenderer((payload) => {
    graph.setGraph(payload.elements);
  });

  // ---------------------------------------------------------------------------
  // 2) Keyword dropdown => fetch subgraph => update Cytoscape
  // ---------------------------------------------------------------------------
  await initKeywordDropdown({
    selectId: "keyword-select",
    defaultKeyword: null, // optionally set e.g. "Transparency"
    onSubgraphFetch: (payload) => {
      // payload.elements should match Cytoscape format: { nodes: [...], edges: [...] }
      graph.setGraph(payload.elements);
    },
  });

  // ---------------------------------------------------------------------------
  // 3) Pipeline upload
  // ---------------------------------------------------------------------------
  wirePipelineRunControls({
    fileInputId: "documents",
    keywordSelectId: "keyword-select",
    runBtnId: "pipeline-run-btn",
    onDone: (result) => {
      renderHumanOversightFromPipelineResult(result);
      // Refresh Compare tab source inventory with the newly processed documents
      refreshComparisonSources();
    },
  });

  wireHumanOversightSubmit({});

  // ---------------------------------------------------------------------------
  // 4) Compare tab (cross-document overlap / alignment / conflicts / ambiguity)
  // ---------------------------------------------------------------------------
  wireComparisonView();

  // ---------------------------------------------------------------------------
  // 5) One-click seed (populate a fresh Neo4j with curated ground truth)
  // ---------------------------------------------------------------------------
  wireSeedButton({ seedBtnId: "seed-graph-btn", keywordSelectId: "keyword-select" });

  // // ---------------------------------------------------------------------------
  // // 4) Optional: graph free-text search (if you keep that UI)
  // // ---------------------------------------------------------------------------
  // wireGraphSearch({
  //   cy: graph.cy, // relies on createCytoscapeGraph exposing cy
  //   searchBtnId: "search-btn",
  //   searchInputId: "node-search",
  //   detailsContentId: "details-content",
  //   endpoint: "/api/search_node",
  // });


  // // ---------------------------------------------------------------------------
  // // 5) Optional: legacy/supervisor oversight logic (keep disabled for now)
  // // ---------------------------------------------------------------------------
  // wireOversightLegacy({
  //   fileInputId: "documents",
  //   verificationSectionId: "oversight-section",
  //   commonVerifiedListId: "common-verified-list",
  //   endpoint: "/api/upload_verify",
  // });
});
