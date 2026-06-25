/**
 * Oversight controller
 * --------------------
 * Renders novelty results into the Oversight tab as an audit-only quality view.
 * Triplet extraction starts automatically from the same normalized review list.
 */

import { startTripletExtractionJob, getJobStatus, startExtendExtractionJob } from "./pipeline_client.js";
import { getExtractionSettings } from "./extraction_settings_controller.js";
import { showOversightOverlay, hideOversightOverlay } from "./oversight_overlay.js";
import { switchToTopLevelTab, TopLevelTabs } from "./utils/tabs.js";
import { renderExtractedTripletsFromJobResult, hasTripletsCache, showCachedTripletsStep } from "./extracted_triplets_controller.js";
import { setOversightStep, OversightSteps } from "./oversight_stepper.js";

const PAGE_SIZE = 10;
const REVIEW_DECISIONS = new Set(["EXISTING", "PARTIALLY_NEW", "NEW"]);

const sortDesc = (a, b) => (b.max_score ?? 0) - (a.max_score ?? 0);

let grouped = null;
let allNoveltyResults = [];
let tripletExtractionInFlight = false;
let tripletExtractionGeneration = 0;

// Human review of quality sentences (before extraction):
//  - deselectedReviewIds: rows the reviewer unchecked (kept visible, not sent)
//  - deletedReviewIds:    rows the reviewer deleted (hidden + not sent)
// A quality is sent to extraction only if it survives the relevance threshold,
// is not deleted, and is not deselected.
const deselectedReviewIds = new Set();
const deletedReviewIds = new Set();

// Live relevance-threshold filtering. Qualities are decomposed once, from the
// chunks that passed the run-time threshold. Dragging the relevance slider can
// accurately FILTER DOWN (drop qualities whose source chunk now scores below the
// threshold); it cannot reveal qualities for chunks that were never decomposed,
// so going below the run threshold only surfaces a "re-run" hint.
let runThreshold = null;      // para_threshold at run time
let currentThreshold = null;  // live threshold from the chunk panel slider

// Auto-extend state: lowering the slider below `extractedFloor` triggers a
// debounced, incremental extraction of the newly-included chunks (decompose +
// classify only the delta). `extractedFloor` tracks the lowest threshold we have
// already extracted down to (starts at the run threshold).
let runJobId = null;
let extractedFloor = null;
let extendInFlight = false;
let extendGeneration = 0;
let autoExtendDisabled = false;   // set if the run context expired (manual fallback)
let pendingExtendThreshold = null; // coalesce drags that happen mid-extend

// Qualities waiting for the user to launch triplet extraction. Extraction is
// NOT auto-fired anymore: the human configures the extraction profile first,
// then clicks "Extract triplets".
let pendingTripletItems = [];

function sourceDocIndex(result) {
  const raw = result?.source_context?.source_doc_index;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : "";
}

function stableReviewId(result) {
  const decision = String(result?.decision || "").trim().toUpperCase();
  const quality = String(result?.quality || "").trim().toLowerCase();
  const sourceIndex = sourceDocIndex(result);
  const neighbor = String(result?.matched_neighbor_sentence || "").trim().toLowerCase();
  return `${decision}::${quality}::${sourceIndex}::${neighbor}`;
}

function normalizeReviewResults(results) {
  const seen = new Set();
  const normalized = [];

  for (const item of results || []) {
    const quality = String(item?.quality || "").trim();
    const decision = String(item?.decision || "").trim().toUpperCase();
    if (!quality || !REVIEW_DECISIONS.has(decision)) continue;

    const reviewItem = {
      ...item,
      quality,
      decision,
      source_context: item?.source_context ?? null,
      max_score: Number.isFinite(Number(item?.max_score)) ? Number(item.max_score) : null,
      matched_neighbor_sentence: String(item?.matched_neighbor_sentence || "").trim(),
      // Source chunk relevance (for live threshold filtering). null score means
      // "always included" (literal match or whole-document mode without scores).
      source_chunk_score: Number.isFinite(Number(item?.source_chunk_score)) ? Number(item.source_chunk_score) : null,
      source_chunk_literal: Boolean(item?.source_chunk_literal),
    };

    const key = stableReviewId(reviewItem);
    if (seen.has(key)) continue;
    seen.add(key);
    normalized.push(reviewItem);
  }

  return normalized;
}

function groupByDecision(results) {
  const filterBy = (decision) => results.filter(r => r.decision === decision).sort(sortDesc);

  return {
    EXISTING: filterBy("EXISTING"),
    PARTIALLY_NEW: filterBy("PARTIALLY_NEW"),
    NEW: filterBy("NEW"),
  };
}

/**
 * Whether a quality survives the current relevance threshold. Literal/synonym
 * matches and qualities without a known chunk score are always kept.
 */
function isResultIncluded(result, threshold) {
  if (threshold == null) return true;
  if (result?.source_chunk_literal) return true;
  const s = result?.source_chunk_score;
  if (s == null || !Number.isFinite(Number(s))) return true;
  return Number(s) >= threshold;
}

/** The qualities currently included under the live threshold. */
function getIncludedResults() {
  return allNoveltyResults.filter(r => isResultIncluded(r, currentThreshold));
}

/** Threshold-included qualities the reviewer has NOT deleted (what we display). */
function getVisibleResults() {
  return getIncludedResults().filter(r => !deletedReviewIds.has(stableReviewId(r)));
}

/** Is this quality currently selected to be sent to extraction / the KG? */
function isReviewSelected(result) {
  const id = stableReviewId(result);
  return !deletedReviewIds.has(id) && !deselectedReviewIds.has(id);
}

/** Visible + selected qualities — the exact set that feeds triplet extraction. */
function getSelectedResults() {
  return getVisibleResults().filter(isReviewSelected);
}

/**
 * Recompute the pending extraction set from the current selection and refresh
 * the "Extract triplets" button + summary. Call after any select/delete change.
 */
function refreshReviewSelection() {
  if (!tripletExtractionInFlight) {
    pendingTripletItems = toTripletExtractionItems(getSelectedResults());
    syncExtractTripletsButton();
  }
  updateQualitySummary();
}

/**
 * React to the relevance slider: re-filter qualities, re-render the decision
 * tables, refresh the pending triplet set, and update the summary. Pure
 * client-side — no LLM, no network. Cheap enough to run on every slider tick,
 * but debounced by the caller.
 */
export function applyRelevanceThresholdToQualities(threshold) {
  const t = Number(threshold);
  currentThreshold = Number.isFinite(t) ? t : null;
  if (!allNoveltyResults.length) return;

  grouped = groupByDecision(getVisibleResults());
  renderDecision("EXISTING", 1);
  renderDecision("PARTIALLY_NEW", 1);
  renderDecision("NEW", 1);
  updateDecisionTabCounts();

  // Only selected (checked, not-deleted) qualities are what "Extract triplets"
  // will process — the threshold pre-filters; the reviewer's checkboxes decide.
  refreshReviewSelection();
}

/** Update the "X of Y qualities included" summary + below-run-threshold hint. */
function updateQualitySummary() {
  const el = document.getElementById("oversight-quality-summary");
  if (!el) return;

  const total = allNoveltyResults.length;
  if (!total || currentThreshold == null) {
    el.classList.add("d-none");
    el.innerHTML = "";
    return;
  }

  const included = getIncludedResults().length;
  const selected = getSelectedResults().length;
  const deleted = deletedReviewIds.size;
  const belowFloor = extractedFloor != null && currentThreshold < extractedFloor - 1e-9;

  const parts = [
    `<i class="bi bi-check2-square me-1"></i><span class="fw-semibold">${selected}</span> selected `,
    `<span class="text-muted">of ${included} included · ${total} total (relevance ≥ ${currentThreshold.toFixed(2)})</span>`,
  ];
  if (deleted > 0) {
    parts.push(` · <span class="text-danger">${deleted} deleted</span>`);
  }

  if (extendInFlight) {
    parts.push(
      ` · <span class="text-primary"><span class="spinner-border spinner-border-sm me-1" `,
      `style="width:.7rem;height:.7rem;border-width:1.5px;" role="status" aria-hidden="true"></span>`,
      `auto-extracting newly included chunks…</span>`
    );
  } else if (autoExtendDisabled && belowFloor) {
    // Auto-extend unavailable (e.g. server restarted and lost the run context).
    parts.push(
      ` · <span class="text-warning-emphasis"><i class="bi bi-exclamation-triangle me-1"></i>`,
      `auto-extract unavailable; re-run to extract qualities from newly included chunks</span>`
    );
  }
  el.className = "small mt-1 text-secondary";
  el.innerHTML = parts.join("");
}

function paginate(items, page, pageSize) {
  const total = items.length;
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const p = Math.min(Math.max(1, page), pages);
  const start = (p - 1) * pageSize;
  return { page: p, pages, slice: items.slice(start, start + pageSize) };
}

function renderTable({ container, items, decisionKey, page }) {
  container.innerHTML = "";

  const { page: cur, pages, slice } = paginate(items, page, PAGE_SIZE);

  const tableWrap = document.createElement("div");
  tableWrap.className = "table-responsive";

  // Selection summary for THIS decision group (across all pages).
  const groupSelectedCount = items.filter(isReviewSelected).length;
  const allSelected = items.length > 0 && groupSelectedCount === items.length;

  const table = document.createElement("table");
  table.className = "table table-hover align-middle mb-2";

  table.innerHTML = `
    <thead>
      <tr>
        <th style="width: 2.5rem;" class="text-center">
          <input type="checkbox" class="form-check-input review-select-all"
            ${allSelected ? "checked" : ""}
            title="Select / deselect all in this tab"
            aria-label="Select all qualities in this tab">
        </th>
        <th>Quality</th>
        <th style="width: 110px;">
          Similarity
          <i
            class="bi bi-info-circle ms-1 text-muted"
            data-bs-toggle="tooltip"
            data-bs-title="Cosine similarity between this quality sentence and the closest existing relation sentence in the retrieved KG subgraph. Higher = closer match."></i>
        </th>
        <th style="width: 3rem;" class="text-center">Del</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;

  const tbody = table.querySelector("tbody");

  slice.forEach(r => {
    const sourceTitle = sourcePopoverTitle(r);
    const sourceContent = sourcePopoverContent(r);
    const selected = isReviewSelected(r);

    const tr = document.createElement("tr");
    if (!selected) tr.classList.add("review-row-deselected");
    tr.innerHTML = `
      <td class="text-center">
        <input type="checkbox" class="form-check-input review-select"
          ${selected ? "checked" : ""} aria-label="Select this quality">
      </td>
      <td class="quality-cell">
        <div class="quality-text d-flex align-items-start gap-2">
          <button type="button"
            class="btn btn-sm btn-link p-0 text-muted"
            data-bs-toggle="popover"
            data-bs-trigger="hover focus"
            data-bs-placement="right"
            data-bs-container="body"
            data-bs-custom-class="source-popover"
            data-bs-title="${escapeHtml(sourceTitle)}"
            data-bs-content="${escapeHtml(sourceContent)}"
            aria-label="${escapeHtml(sourceTitle)}">
            <i class="bi bi-info-circle"></i>
          </button>
          <span class="flex-grow-1">${escapeHtml(r.quality)}</span>
        </div>
        ${dimensionBadges(r)}
      </td>
      <td>
        <span class="badge text-bg-secondary">${formatSimilarity(r.max_score)}</span>
      </td>
      <td class="text-center">
        <button type="button" class="btn btn-sm btn-link p-0 text-danger review-delete"
          title="Delete this quality (won't be sent to the KG)" aria-label="Delete this quality">
          <i class="bi bi-trash"></i>
        </button>
      </td>
    `;

    const cb = tr.querySelector(".review-select");
    cb.addEventListener("change", () => {
      const id = stableReviewId(r);
      if (cb.checked) deselectedReviewIds.delete(id);
      else deselectedReviewIds.add(id);
      tr.classList.toggle("review-row-deselected", !cb.checked);
      // keep the tab's select-all box in sync
      const head = table.querySelector(".review-select-all");
      if (head) head.checked = items.every(isReviewSelected);
      refreshReviewSelection();
    });

    tr.querySelector(".review-delete").addEventListener("click", () => {
      deletedReviewIds.add(stableReviewId(r));
      grouped = groupByDecision(getVisibleResults());
      renderDecision(decisionKey, cur);
      refreshReviewSelection();
    });

    tbody.appendChild(tr);
  });

  // Header select-all toggles every row in this decision group (all pages).
  const selectAll = table.querySelector(".review-select-all");
  if (selectAll) {
    selectAll.addEventListener("change", () => {
      items.forEach(r => {
        const id = stableReviewId(r);
        if (selectAll.checked) deselectedReviewIds.delete(id);
        else deselectedReviewIds.add(id);
      });
      renderDecision(decisionKey, cur);
      refreshReviewSelection();
    });
  }

  tableWrap.appendChild(table);
  container.appendChild(tableWrap);

  const pager = document.createElement("div");
  pager.className = "d-flex justify-content-between align-items-center";

  const left = document.createElement("div");
  left.className = "text-muted small";
  left.textContent = `${items.length} total • page ${cur}/${pages}`;

  const right = document.createElement("div");
  right.className = "btn-group btn-group-sm";

  const prev = document.createElement("button");
  prev.className = "btn btn-outline-secondary";
  prev.textContent = "Prev";
  prev.disabled = cur <= 1;
  prev.addEventListener("click", () => {
    renderDecision(decisionKey, cur - 1);
  });

  const next = document.createElement("button");
  next.className = "btn btn-outline-secondary";
  next.textContent = "Next";
  next.disabled = cur >= pages;
  next.addEventListener("click", () => {
    renderDecision(decisionKey, cur + 1);
  });

  right.appendChild(prev);
  right.appendChild(next);
  pager.appendChild(left);
  pager.appendChild(right);
  container.appendChild(pager);

  initBootstrapHints(container);
}

function renderDecision(decisionKey, page = 1) {
  const pane = document.querySelector(`#oversight-${decisionKey.toLowerCase()} .qualities-container`);
  if (!pane || !grouped) return;
  renderTable({ container: pane, items: grouped[decisionKey], decisionKey, page });
}

// The three review tabs, in display order.
const DECISION_TABS = [
  { key: "EXISTING", id: "oversight-existing-tab", label: "Existing Knowledge" },
  { key: "PARTIALLY_NEW", id: "oversight-partially_new-tab", label: "Partial Match" },
  { key: "NEW", id: "oversight-new-tab", label: "New Knowledge" },
];

/** Show "<label> (n)" on each tab so the reviewer sees where the lines are. */
function updateDecisionTabCounts() {
  if (!grouped) return;
  for (const t of DECISION_TABS) {
    const btn = document.getElementById(t.id);
    if (!btn) continue;
    const n = (grouped[t.key] || []).length;
    btn.innerHTML = `${t.label} <span class="badge rounded-pill ${n ? "text-bg-secondary" : "text-bg-light text-muted"} ms-1">${n}</span>`;
  }
}

/** Switch to the first tab that actually has rows (so the reviewer never lands
 *  on an empty tab and thinks there are no lines / no checkboxes). */
function activateFirstPopulatedDecisionTab() {
  if (!grouped || !window.bootstrap?.Tab) return;
  const target = DECISION_TABS.find(t => (grouped[t.key] || []).length > 0);
  if (!target) return;
  const btn = document.getElementById(target.id);
  if (btn) window.bootstrap.Tab.getOrCreateInstance(btn).show();
}

function getTripletStatusEl() {
  return document.getElementById("oversight-triplet-status");
}

function setTripletStatus(message) {
  const el = getTripletStatusEl();
  if (el) el.textContent = message;
}

function summarizeTripletJobResult(jobResult, qualitiesSent) {
  const extracted = Array.isArray(jobResult?.extracted_triplets) ? jobResult.extracted_triplets : [];
  const tripletRows = extracted.reduce((total, item) => {
    const triplets = Array.isArray(item?.triplets) ? item.triplets : [];
    return total + triplets.length;
  }, 0);
  const skipped = extracted.filter(item => {
    const triplets = Array.isArray(item?.triplets) ? item.triplets : [];
    return triplets.length === 0;
  }).length;

  return `Triplet extraction: ${qualitiesSent} qualities sent, ${tripletRows} triplet rows, ${skipped} skipped/no-fit.`;
}

/**
 * Public entrypoint: call this when pipeline finishes.
 */
export function renderHumanOversightFromPipelineResult(pipelineResult, runContext = {}) {
  const novelty = pipelineResult?.NoveltyLLM;
  const results = Array.isArray(novelty?.results) ? novelty.results : [];
  allNoveltyResults = normalizeReviewResults(results);
  // Fresh review session: every quality starts selected and none deleted.
  deselectedReviewIds.clear();
  deletedReviewIds.clear();

  // Threshold the qualities were extracted at; the slider starts here so the
  // initial view shows every extracted quality.
  const para = Number(pipelineResult?.KeyBERT?.para_threshold);
  runThreshold = Number.isFinite(para) ? para : null;
  currentThreshold = runThreshold;

  // Auto-extend bookkeeping: we have extracted down to the run threshold so far.
  runJobId = runContext?.jobId || null;
  extractedFloor = runThreshold;
  extendInFlight = false;
  autoExtendDisabled = false;
  pendingExtendThreshold = null;
  ++extendGeneration;

  grouped = groupByDecision(getVisibleResults());

  renderDecision("EXISTING", 1);
  renderDecision("PARTIALLY_NEW", 1);
  renderDecision("NEW", 1);
  updateDecisionTabCounts();
  activateFirstPopulatedDecisionTab();

  switchToTopLevelTab({ tab: TopLevelTabs.OVERSIGHT });
  setOversightStep(OversightSteps.CANDIDATE_SENTENCES);

  syncGoToTripletsButton();
  wireGoToTripletsButton();

  const selectedItems = toTripletExtractionItems(getSelectedResults());
  ++tripletExtractionGeneration;
  pendingTripletItems = selectedItems;
  updateQualitySummary();

  if (!selectedItems.length) {
    setTripletStatus("Triplet extraction: no reviewed qualities available.");
    syncExtractTripletsButton();
    return;
  }

  // Do NOT auto-extract. The human-in-the-loop reviews qualities and confirms
  // the extraction profile (predicate families, edge mode, modality, custom
  // predicates) first, then clicks "Extract triplets". The profile can be set
  // while the pipeline is still running.
  setTripletStatus(
    `${selectedItems.length} qualities ready. Review your extraction profile above, then click “Extract triplets”.`
  );
  syncExtractTripletsButton();
}

function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatSimilarity(score) {
  return Number.isFinite(score) ? score.toFixed(2) : "n/a";
}

/**
 * Render the dimension tag(s) a quality belongs to as small pills. A single
 * quality can match several trustworthy-AI dimensions (complete scan).
 */
function dimensionBadges(result) {
  const dims = Array.isArray(result?.dimensions) ? result.dimensions.filter(Boolean) : [];
  if (!dims.length) return "";
  const pills = dims
    .map(d => `<span class="badge rounded-pill text-bg-info" style="font-weight:500;">${escapeHtml(d)}</span>`)
    .join(" ");
  return `<div class="d-flex flex-wrap gap-1 mt-1 ms-4"><i class="bi bi-tag text-muted" style="font-size:0.75rem;"></i>${pills}</div>`;
}

function sourcePopoverTitle(result) {
  const rawIndex = sourceDocIndex(result);
  if (Number.isInteger(rawIndex)) {
    return `Source paragraph ${rawIndex + 1}`;
  }
  return "Source paragraph";
}

function sourcePopoverContent(result) {
  const ctx = result?.source_context;
  const text = String(ctx?.source_text || "").trim();

  if (!text) {
    return "No source paragraph was attached to this quality.";
  }

  const metadata = ctx?.metadata || {};
  const headings = Array.isArray(metadata?.headings)
    ? metadata.headings.map(String).filter(Boolean)
    : [];
  const docName = String(ctx?.doc_name || metadata?.source || "").trim();

  const parts = [];
  if (docName) {
    parts.push(`Document: ${docName}`);
  }
  if (headings.length > 0) {
    parts.push(headings.join(" > "));
  }
  parts.push(text);

  return parts.join("\n\n");
}

function toTripletExtractionItems(results) {
  return (results || [])
    .map(r => ({
      quality: String(r?.quality ?? "").trim(),
      source_context: r?.source_context ?? null,
      decision: r?.decision ?? null,
      max_score: r?.max_score ?? null,
      matched_neighbor_sentence: r?.matched_neighbor_sentence ?? null,
      dimensions: Array.isArray(r?.dimensions) ? r.dimensions : [],
    }))
    .filter(item => item.quality);
}

async function runTripletExtractionJobForItems(selectedItems, {
  overlayTitle = "Extracting triplets...",
  overlaySubtitle = "Applying allowed predicates to qualities.",
  confirmOverwrite = false,
  showOverlay = true,
  activateOnDone = true,
  alertOnError = true,
  pollIntervalMs = 1000,
  generation = tripletExtractionGeneration,
} = {}) {
  if (!selectedItems.length) {
    if (alertOnError) alert("No quality text was available for triplet extraction.");
    return false;
  }

  if (tripletExtractionInFlight) {
    if (alertOnError) alert("Triplet extraction is already running in the background.");
    return false;
  }

  tripletExtractionInFlight = true;
  syncExtractTripletsButton();
  setTripletStatus(`Triplet extraction: ${selectedItems.length} qualities sent, waiting for LLM results...`);
  if (showOverlay) showOversightOverlay(overlayTitle, overlaySubtitle);

  try {
    const start = await startTripletExtractionJob({
      selected_items: selectedItems,
      extraction_settings: getExtractionSettings(),
    });
    const jobId = start.job_id;

    while (true) {
      const job = await getJobStatus(jobId);
      if (job.state === "done") {
        if (generation !== tripletExtractionGeneration) return false;
        renderExtractedTripletsFromJobResult(job.result, { activate: activateOnDone });
        setTripletStatus(summarizeTripletJobResult(job.result, selectedItems.length));
        syncGoToTripletsButton();
        return true;
      }
      if (job.state === "error") {
        throw new Error(job.error || "Triplet extraction failed.");
      }
      await sleep(pollIntervalMs);
    }
  } catch (e) {
    console.error("Triplet extraction failed:", e);
    setTripletStatus(`Triplet extraction failed: ${e.message || String(e)}`);
    if (alertOnError) alert(e.message || String(e));
    return false;
  } finally {
    tripletExtractionInFlight = false;
    syncExtractTripletsButton();
    if (showOverlay) hideOversightOverlay();
  }
}

export function wireHumanOversightSubmit() {
  wireGoToTripletsButton();
  wireExtractTripletsButton();
  wireRelevanceThresholdSync();
}

/**
 * Listen for the chunk-relevance slider (in the chunk-scores panel):
 *   - filter the already-extracted qualities instantly (90ms debounce), and
 *   - auto-extend extraction to newly-included chunks once the drag settles
 *     (700ms debounce) — this is the LLM step, so it only fires when the user
 *     stops moving the slider.
 */
let filterDebounce = null;
let extendDebounce = null;
function wireRelevanceThresholdSync() {
  if (document.body.dataset.relevanceSyncWired) return;
  document.body.dataset.relevanceSyncWired = "1";

  document.addEventListener("kb:chunk-threshold-change", (e) => {
    const threshold = e?.detail?.threshold;

    // Instant client-side filtering (cheap).
    if (filterDebounce) clearTimeout(filterDebounce);
    filterDebounce = setTimeout(() => applyRelevanceThresholdToQualities(threshold), 90);

    // Debounced incremental extraction (LLM) once dragging settles.
    if (extendDebounce) clearTimeout(extendDebounce);
    extendDebounce = setTimeout(() => maybeAutoExtend(threshold), 700);
  });
}

/** Build the merge-key set so new results don't duplicate existing ones. */
function mergeNewResults(newResults) {
  const existingKeys = new Set(allNoveltyResults.map(stableReviewId));
  let added = 0;
  for (const r of newResults) {
    const key = stableReviewId(r);
    if (existingKeys.has(key)) continue;
    existingKeys.add(key);
    allNoveltyResults.push(r);
    added += 1;
  }
  return added;
}

/**
 * If the threshold dropped below what we've already extracted, kick off a
 * debounced incremental extraction for the newly-included chunks.
 */
function maybeAutoExtend(threshold) {
  const t = Number(threshold);
  if (!Number.isFinite(t)) return;
  if (!runJobId || autoExtendDisabled) return;          // no context / gave up
  if (extractedFloor == null || t >= extractedFloor - 1e-9) return; // nothing new below floor

  if (extendInFlight) {
    // Remember the lowest threshold requested while busy; run it after.
    pendingExtendThreshold = pendingExtendThreshold == null ? t : Math.min(pendingExtendThreshold, t);
    return;
  }
  void runAutoExtend(t);
}

async function runAutoExtend(threshold) {
  extendInFlight = true;
  const generation = extendGeneration;
  updateQualitySummary();  // shows the "extracting…" state

  try {
    const start = await startExtendExtractionJob({ source_job_id: runJobId, threshold });
    const jobId = start?.job_id;
    if (!jobId) throw new Error("No job id returned for extend extraction.");

    let result = null;
    while (true) {
      const job = await getJobStatus(jobId);
      if (job.state === "done") { result = job.result; break; }
      if (job.state === "error") throw new Error(job.error || "Extend extraction failed.");
      await sleep(800);
    }

    // A newer run replaced this one while we waited — discard.
    if (generation !== extendGeneration) return;

    const newResults = normalizeReviewResults(result?.results || []);
    const added = mergeNewResults(newResults);

    // We've now extracted down to this threshold.
    extractedFloor = Math.min(extractedFloor ?? threshold, threshold);

    // Re-group + re-render at whatever the current slider value is.
    applyRelevanceThresholdToQualities(currentThreshold);

    if (added > 0) {
      setTripletStatus(`Auto-extracted ${added} new qualities from chunks above relevance ${threshold.toFixed(2)}.`);
    }
  } catch (e) {
    console.error("Auto-extend failed:", e);
    // Stop hammering; fall back to the manual re-run hint.
    autoExtendDisabled = true;
  } finally {
    extendInFlight = false;
    updateQualitySummary();
    // Coalesced lower request arrived during this extend — process it.
    if (!autoExtendDisabled && pendingExtendThreshold != null) {
      const next = pendingExtendThreshold;
      pendingExtendThreshold = null;
      if (extractedFloor != null && next < extractedFloor - 1e-9) void runAutoExtend(next);
    }
  }
}

function sleep(ms) {
  return new Promise(res => setTimeout(res, ms));
}

const getGoToTripletsButton = () => document.getElementById("oversight-go-triplets");

function syncGoToTripletsButton() {
  const btn = getGoToTripletsButton();
  if (!btn) return;
  const hasCache = hasTripletsCache();
  btn.classList.toggle("d-none", !hasCache);
  btn.classList.toggle("btn-primary", hasCache);
  btn.classList.toggle("btn-outline-secondary", !hasCache);
}

export function wireGoToTripletsButton() {
  const btn = getGoToTripletsButton();
  if (!btn || btn.dataset.wired) return;
  btn.dataset.wired = "1";

  btn.addEventListener("click", () => {
    const ok = showCachedTripletsStep();
    if (!ok) {
      alert("No extracted triplets yet. Click “Extract triplets” to generate them with your chosen profile.");
    }
  });
}

const getExtractTripletsButton = () => document.getElementById("oversight-extract-triplets");

/**
 * Show the "Extract triplets" call-to-action when qualities are waiting and
 * extraction is not already running.
 */
function syncExtractTripletsButton() {
  const btn = getExtractTripletsButton();
  if (!btn) return;
  const ready = pendingTripletItems.length > 0 && !tripletExtractionInFlight;
  btn.classList.toggle("d-none", !ready);
  btn.disabled = !ready;
  const countEl = document.getElementById("extract-triplets-count");
  if (countEl) countEl.textContent = ready ? ` (${pendingTripletItems.length} selected)` : "";
}

export function wireExtractTripletsButton() {
  const btn = getExtractTripletsButton();
  if (!btn || btn.dataset.wired) return;
  btn.dataset.wired = "1";

  btn.addEventListener("click", () => {
    if (!pendingTripletItems.length) {
      alert("No reviewed qualities are available for extraction yet.");
      return;
    }
    const generation = ++tripletExtractionGeneration;
    syncExtractTripletsButton();
    void runTripletExtractionJobForItems(pendingTripletItems, {
      generation,
      showOverlay: true,
      activateOnDone: true,
      alertOnError: true,
      pollIntervalMs: 1500,
      confirmOverwrite: false,
    });
  });
}

/**
 * Reset the Human Oversight UI so a new run starts clean.
 */
export function resetHumanOversightUI() {
  allNoveltyResults = [];
  grouped = null;
  deselectedReviewIds.clear();
  deletedReviewIds.clear();
  runThreshold = null;
  currentThreshold = null;
  runJobId = null;
  extractedFloor = null;
  extendInFlight = false;
  autoExtendDisabled = false;
  pendingExtendThreshold = null;
  ++extendGeneration;
  tripletExtractionInFlight = false;
  tripletExtractionGeneration += 1;
  pendingTripletItems = [];
  syncExtractTripletsButton();
  setTripletStatus("Triplet extraction: waiting for reviewed qualities.");
  updateQualitySummary();

  ["existing", "partially_new", "new"].forEach((k) => {
    const pane = document.querySelector(`#oversight-${k} .qualities-container`);
    if (pane) pane.innerHTML = "";
  });

  const bottom = document.getElementById("oversight-bottom");
  if (bottom) bottom.classList.add("d-none");

  syncGoToTripletsButton();
}

/**
 * Whether the Candidate Sentences UI currently has any rendered items.
 *
 * @returns {boolean}
 */
export function hasCandidateQualitiesUI() {
  const containers = document.querySelectorAll(
    "#oversight-existing .qualities-container, #oversight-partially_new .qualities-container, #oversight-new .qualities-container"
  );

  for (const el of containers) {
    if (el && el.children && el.children.length > 0) return true;
    if (el && el.textContent && el.textContent.trim().length > 0) return true;
  }
  return false;
}

function initBootstrapHints(root) {
  if (!window.bootstrap) return;

  root.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
    new bootstrap.Tooltip(el);
  });

  root.querySelectorAll('[data-bs-toggle="popover"]').forEach(el => {
    new bootstrap.Popover(el);
  });
}
