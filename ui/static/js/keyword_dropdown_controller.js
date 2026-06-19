/**
 * Keyword dropdown controller:
 * - loads keywords from /api/search-keywords
 * - populates <select>
 * - does NOT auto-select any keyword
 * - enables upload ONLY after a keyword is selected
 * - on change: fetches /api/subgraph and updates the graph
 *
 * This module owns upload enablement because it is currently 100% driven
 * by the dropdown's state (selected keyword + loading state).
 */

import { getSearchKeywords, getSubgraph, getFullGraph } from "./graph_client.js";
import { switchToTopLevelTab, TopLevelTabs } from "./utils/tabs.js"
import { showGlobalLoading, hideGlobalLoading } from "./modals/global_loading_modal.js";
import { setLastSubgraphPayload } from "./state/graph_state.js";

let currentSubgraphAbort = null;

/**
 * Sentinel value for the "complete scan" option (all trustworthy-AI dimensions).
 * UI-only for now; the backend scan-all path is wired later.
 */
export const ALL_DIMENSIONS = "__ALL_DIMENSIONS__";

/** Sentinel: user types their own keyword(s) (free text). */
export const CUSTOM_KEYWORD = "__CUSTOM__";

/** Sentinel: no keyword — extract the whole document (text → graph). */
export const NO_KEYWORD = "__NO_KEYWORD__";

/**
 * Resolve the current keyword selection into the payload the pipeline run needs.
 * @returns {{keyword: string, custom_keywords: string}}
 */
export function getKeywordSelection() {
  const select = document.getElementById("keyword-select");
  const value = (select?.value || "").trim();
  if (value === CUSTOM_KEYWORD) {
    const input = document.getElementById("custom-keyword-input");
    return { keyword: CUSTOM_KEYWORD, custom_keywords: (input?.value || "").trim() };
  }
  return { keyword: value, custom_keywords: "" };
}

/**
 * Whether the current selection is enough to enable Run:
 * - NO_KEYWORD: always ready (whole document)
 * - CUSTOM: ready only when the custom input has text
 * - dimension / complete scan: ready when a value is chosen
 */
export function isKeywordSelectionReady() {
  const sel = getKeywordSelection();
  if (!sel.keyword) return false;
  if (sel.keyword === CUSTOM_KEYWORD) return sel.custom_keywords.length > 0;
  return true;
}

/**
 * Initialize the keyword dropdown and connect it to subgraph fetch/render.
 *
 * @param {Object} params
 * @param {string} params.selectId - The <select> element id.
 * @param {Function} params.onSubgraphFetch - Callback invoked with the Cytoscape payload.
 * @param {boolean} [params.useGlobalOverlay=false] - Whether to use the global loading overlay.
 * @param {string} [params.fileInputId="documents"] - File input to enable/disable based on keyword selection.
 */
export async function initKeywordDropdown({
  selectId = "keyword-select",
  onSubgraphFetch,
  useGlobalOverlay = false,
  fileInputId = "documents",
}) {
  const select = document.getElementById(selectId);
  if (!select) throw new Error(`Missing select element #${selectId}`);

  const fileInput = document.getElementById(fileInputId);
  const keywordSpinner = document.getElementById("keyword-spinner");

  /**
   * Local UI state:
   * - hasKeywordSelected: user has chosen a real keyword
   * - isLoading: currently fetching keywords or subgraph
   *
   * Upload should be enabled ONLY if:
   *   hasKeywordSelected && !isLoading
   */
  let hasKeywordSelected = false;
  let isLoading = false;

  /**
   * Compute + apply UI state in one place (prevents bugs like "stuck disabled").
   */
  function syncUI() {
    // Disable dropdown during loading to avoid multi-clicks and request racing
    select.disabled = isLoading;

    // Toggle spinner near keyword dropdown
    if (keywordSpinner) keywordSpinner.classList.toggle("d-none", !isLoading);

    // Upload input is enabled ONLY when keyword chosen AND not loading
    if (fileInput) fileInput.disabled = !hasKeywordSelected || isLoading;

    // // Optional global overlay
    // if (useGlobalOverlay && overlay) {
    //   overlay.classList.toggle("d-none", !isLoading);
    // }
  }

  /**
   * Set loading state + optionally set overlay text.
   */
  function setLoading(nextLoading, { title = "Loading…", subtitle = "Please wait." } = {}) {
    isLoading = nextLoading;

    if (useGlobalOverlay) {
      if (isLoading) showGlobalLoading(title, subtitle);
      else hideGlobalLoading();
    }

    syncUI();
  }

  /**
   * Helper to set whether a valid keyword is selected.
   */
  function setHasKeywordSelected(selected) {
    hasKeywordSelected = selected;
    syncUI();
  }

  // Default: upload disabled until keyword chosen
  setHasKeywordSelected(false);

  try {
    setLoading(true, {
      title: "Loading focus areas…",
      subtitle: "Populating the trustworthy-AI dimensions.",
    });

    const data = await getSearchKeywords();
    const keywords = data.keywords || [];

    // Build select options
    select.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select a focus area…";
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);

    // Complete-scan option (all dimensions) — listed first for visibility.
    const allOpt = document.createElement("option");
    allOpt.value = ALL_DIMENSIONS;
    allOpt.textContent = "🔍 Complete trustworthy-AI scan (all dimensions)";
    select.appendChild(allOpt);

    // Custom keyword(s) — user supplies their own focus terms.
    const customOpt = document.createElement("option");
    customOpt.value = CUSTOM_KEYWORD;
    customOpt.textContent = "✏️ Custom keyword(s)…";
    select.appendChild(customOpt);

    // No keyword — whole-document extraction (text → graph).
    const noneOpt = document.createElement("option");
    noneOpt.value = NO_KEYWORD;
    noneOpt.textContent = "📄 No keyword — whole document (text → graph)";
    select.appendChild(noneOpt);

    const dimGroup = document.createElement("optgroup");
    dimGroup.label = "Single dimension";
    for (const k of keywords) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k;
      dimGroup.appendChild(opt);
    }
    select.appendChild(dimGroup);

    // IMPORTANT: no auto-selection, no auto-fetch.

    const customInput = document.getElementById("custom-keyword-input");
    const showCustomInput = (show) => {
      if (customInput) customInput.classList.toggle("d-none", !show);
    };

    // Enable Run as the user types custom keyword(s).
    if (customInput) {
      customInput.addEventListener("input", () => {
        if ((select.value || "").trim() === CUSTOM_KEYWORD) {
          setHasKeywordSelected(customInput.value.trim().length > 0);
        }
      });
    }

    select.addEventListener("change", async () => {
      const chosen = (select.value || "").trim();

      // If somehow empty, keep upload disabled
      if (!chosen) {
        showCustomInput(false);
        setHasKeywordSelected(false);
        return;
      }

      // Custom keyword(s): reveal the text box; Run enables once it has text.
      if (chosen === CUSTOM_KEYWORD) {
        showCustomInput(true);
        setHasKeywordSelected((customInput?.value || "").trim().length > 0);
        if (customInput) customInput.focus();
        return;
      }

      showCustomInput(false);

      // No keyword: whole-document extraction. Nothing to filter, no subgraph to
      // preview — just enable Run.
      if (chosen === NO_KEYWORD) {
        setHasKeywordSelected(true);
        return;
      }

      // Complete scan: preview the WHOLE graph from Neo4j AND enable Run so the
      // pipeline can extract across every dimension in one pass. Each quality is
      // tagged with the dimension(s) it matches.
      if (chosen === ALL_DIMENSIONS) {
        setHasKeywordSelected(true);  // allow upload + Run for a complete scan
        setLoading(true, {
          title: "Loading full graph…",
          subtitle: "Fetching all nodes and relationships from Neo4j.",
        });
        try {
          await fetchAndRenderFullGraph(onSubgraphFetch);
          switchToTopLevelTab({ tab: TopLevelTabs.GRAPH });
        } finally {
          setLoading(false);
        }
        return;
      }

      // Mark keyword as chosen (upload would become enabled once not loading)
      setHasKeywordSelected(true);

      setLoading(true, {
        title: "Loading graph…",
        subtitle: "If the database is waking up, this may take a moment.",
      });

      try {
        await fetchAndRenderSubgraph(chosen, onSubgraphFetch);
        switchToTopLevelTab({ tab: TopLevelTabs.GRAPH });
      } finally {
        setLoading(false);
        // Upload will automatically be enabled because:
        // hasKeywordSelected === true and isLoading === false
      }
    });
  } catch (err) {
    console.error(err);
    select.innerHTML = `<option value="">❌ Failed to load focus areas</option>`;
    select.disabled = true;
    setHasKeywordSelected(false);
  } finally {
    // Ensure spinner/overlay cleared
    setLoading(false);
  }
}

/**
 * Fetch subgraph for keyword and hand it to the caller for rendering.
 * Uses AbortController to cancel any in-flight request when user changes keyword quickly.
 *
 * @param {string} keyword
 * @param {Function} onSubgraphFetch
 */
async function fetchAndRenderSubgraph(keyword, onSubgraphFetch) {
  try {
    if (currentSubgraphAbort) currentSubgraphAbort.abort();
    currentSubgraphAbort = new AbortController();

    const subgraphPayload = await getSubgraph(keyword, {
      signal: currentSubgraphAbort.signal,
    });

    // ✅ cache it for export
    setLastSubgraphPayload(subgraphPayload);

    onSubgraphFetch(subgraphPayload);
  } catch (err) {
    if (err?.name === "AbortError") return;
    console.error(err);
    alert(`Subgraph error: ${err.message}`);
  } finally {
    currentSubgraphAbort = null;
  }
}

/**
 * Fetch the entire graph (all nodes + relationships) and render it.
 * Used by the "Complete scan (all dimensions)" option.
 *
 * @param {Function} onSubgraphFetch
 */
async function fetchAndRenderFullGraph(onSubgraphFetch) {
  try {
    if (currentSubgraphAbort) currentSubgraphAbort.abort();
    currentSubgraphAbort = new AbortController();

    const payload = await getFullGraph({ signal: currentSubgraphAbort.signal });

    setLastSubgraphPayload(payload);
    onSubgraphFetch(payload);
  } catch (err) {
    if (err?.name === "AbortError") return;
    console.error(err);
    alert(`Full graph error: ${err.message}`);
  } finally {
    currentSubgraphAbort = null;
  }
}