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
import { initExtractionSettings } from "./extraction_settings_controller.js";
import { initPipelineTuning } from "./pipeline_tuning_controller.js";
import { renderChunkScoresDebug } from "./chunk_scores_debug.js"; // DEBUG/TEST (safe to remove)
import { wireSeedButton } from "./seed_controller.js";
import { registerSubgraphRenderer } from "./graph_refresh.js";
import { hideProgressPanel } from "./pipeline_progress_ui.js"
import { initVerificationPanel } from "./verification_controller.js";

// Optional modules (TODO: Most likely will be deleted)
// import { wireGraphSearch } from "./graph_free_text_search.js";

function initSettingsSidebar() {
  const sidebar = document.getElementById("oversight-settings-sidebar");
  const toggleBtn = document.getElementById("settings-sidebar-toggle");
  const icon = document.getElementById("sidebar-toggle-icon");
  const strip = sidebar?.querySelector(".sidebar-collapsed-strip");
  if (!sidebar || !toggleBtn) return;

  const LS_KEY = "kb.settings.sidebar.state";

  const applyState = (collapsed) => {
    sidebar.classList.toggle("sidebar-collapsed", collapsed);
    if (icon) icon.className = `bi ${collapsed ? "bi-arrow-bar-right" : "bi-arrow-bar-left"}`;
    localStorage.setItem(LS_KEY, collapsed ? "1" : "0");
  };

  applyState(localStorage.getItem(LS_KEY) === "1");

  toggleBtn.addEventListener("click", () => applyState(!sidebar.classList.contains("sidebar-collapsed")));
  strip?.addEventListener("click", () => applyState(false));
}

document.addEventListener("DOMContentLoaded", async () => {
  initSettingsSidebar();
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
    onDone: (result, ctx) => {
      renderHumanOversightFromPipelineResult(result, ctx);
      // Refresh Compare tab source inventory with the newly processed documents
      refreshComparisonSources();
      // DEBUG/TEST (safe to remove): show per-chunk KeyBERT scores.
      renderChunkScoresDebug(result);
    },
  });

  wireHumanOversightSubmit({});

  // Extraction profile (predicate families, edge-label policy, modality, custom predicates)
  initExtractionSettings({ containerId: "extraction-settings" });

  // Tunable pipeline thresholds (paragraph relevance, KG similarity, top-k, limit)
  initPipelineTuning({ containerId: "pipeline-tuning" });

  // "Tune thresholds" toolbar shortcut → jump to Review tab, open sidebar, expand thresholds
  document.getElementById("jump-to-thresholds")?.addEventListener("click", () => {
    document.getElementById("oversight-view-tab")?.click();
    const sidebar = document.getElementById("oversight-settings-sidebar");
    if (sidebar?.classList.contains("sidebar-collapsed")) {
      document.getElementById("settings-sidebar-toggle")?.click();
    }
    const section = document.getElementById("sb-pipeline-section");
    if (section && !section.classList.contains("show")) {
      window.bootstrap?.Collapse?.getOrCreateInstance(section)?.show();
    }
    // Scroll threshold panel into view after expand animation
    setTimeout(() => document.getElementById("pipeline-tuning")?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 320);
  });

  // ---------------------------------------------------------------------------
  // 4) Compare tab (cross-document overlap / alignment / conflicts / ambiguity)
  // ---------------------------------------------------------------------------
  wireComparisonView();

  // ---------------------------------------------------------------------------
  // 5) One-click seed (populate a fresh Neo4j with curated ground truth)
  // ---------------------------------------------------------------------------
  wireSeedButton({ seedBtnId: "seed-graph-btn", keywordSelectId: "keyword-select" });

  // ---------------------------------------------------------------------------
  // 6) Phase B verification panel (auto-runs after KG upsert; manual re-run)
  // ---------------------------------------------------------------------------
  initVerificationPanel();

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
