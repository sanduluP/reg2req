/**
 * Oversight controller
 * --------------------
 * Renders novelty results into the Oversight tab as an audit-only quality view.
 * Triplet extraction starts automatically from the same normalized review list.
 */

import { startTripletExtractionJob, getJobStatus } from "./pipeline_client.js";
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

  const table = document.createElement("table");
  table.className = "table table-hover align-middle mb-2";

  table.innerHTML = `
    <thead>
      <tr>
        <th>Quality</th>
        <th style="width: 130px;">
          Similarity
          <i
            class="bi bi-info-circle ms-1 text-muted"
            data-bs-toggle="tooltip"
            data-bs-title="Cosine similarity between this quality sentence and the closest existing relation sentence in the retrieved KG subgraph. Higher = closer match."></i>
        </th>
      </tr>
    </thead>
    <tbody></tbody>
  `;

  const tbody = table.querySelector("tbody");

  slice.forEach(r => {
    const sourceTitle = sourcePopoverTitle(r);
    const sourceContent = sourcePopoverContent(r);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="quality-cell">
        <div class="fw-semibold fs-6 d-flex align-items-start gap-2">
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
      </td>
      <td>
        <span class="badge text-bg-secondary">${formatSimilarity(r.max_score)}</span>
      </td>
    `;

    tbody.appendChild(tr);
  });

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
export function renderHumanOversightFromPipelineResult(pipelineResult) {
  const novelty = pipelineResult?.NoveltyLLM;
  const results = Array.isArray(novelty?.results) ? novelty.results : [];
  allNoveltyResults = normalizeReviewResults(results);
  grouped = groupByDecision(allNoveltyResults);

  renderDecision("EXISTING", 1);
  renderDecision("PARTIALLY_NEW", 1);
  renderDecision("NEW", 1);

  switchToTopLevelTab({ tab: TopLevelTabs.OVERSIGHT });
  setOversightStep(OversightSteps.CANDIDATE_SENTENCES);

  syncGoToTripletsButton();
  wireGoToTripletsButton();

  const selectedItems = toTripletExtractionItems(allNoveltyResults);
  const generation = ++tripletExtractionGeneration;

  if (!selectedItems.length) {
    setTripletStatus("Triplet extraction: no reviewed qualities available.");
    return;
  }

  setTripletStatus(`Triplet extraction: ${selectedItems.length} qualities queued for predicate-constrained extraction.`);

  const startBackgroundTriplets = () => {
    void runTripletExtractionJobForItems(selectedItems, {
      generation,
      showOverlay: false,
      activateOnDone: false,
      alertOnError: false,
      pollIntervalMs: 2000,
      confirmOverwrite: false,
    });
  };

  if ("requestIdleCallback" in window) {
    window.requestIdleCallback(startBackgroundTriplets, { timeout: 1000 });
  } else {
    setTimeout(startBackgroundTriplets, 0);
  }
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
  setTripletStatus(`Triplet extraction: ${selectedItems.length} qualities sent, waiting for LLM results...`);
  if (showOverlay) showOversightOverlay(overlayTitle, overlaySubtitle);

  try {
    const start = await startTripletExtractionJob({ selected_items: selectedItems });
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
    if (showOverlay) hideOversightOverlay();
  }
}

export function wireHumanOversightSubmit() {
  wireGoToTripletsButton();
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
      alert("No extracted triplets yet. Triplets are generated automatically after a pipeline run.");
    }
  });
}

/**
 * Reset the Human Oversight UI so a new run starts clean.
 */
export function resetHumanOversightUI() {
  allNoveltyResults = [];
  grouped = null;
  tripletExtractionInFlight = false;
  tripletExtractionGeneration += 1;
  setTripletStatus("Triplet extraction: waiting for reviewed qualities.");

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
