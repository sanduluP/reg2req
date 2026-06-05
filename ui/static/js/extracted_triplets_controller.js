/**
 * Extracted Triplets Controller
 * -----------------------------
 * Responsible for rendering and managing the "Extracted Triplets" review UI:
 * - Shows/hides the correct Oversight sections
 * - Flattens pipeline extraction results into editable rows
 * - Inline editing for (subject, predicate, object)
 * - Allows deleting rows (soft delete)
 * - Provides submit-to-KG payload of cleaned triplets
 *
 * Expected server result shape (from job.result):
 * {
 *   extracted_triplets: [
 *     { sentence: string, triplets: [[subj, obj, pred], ...] }  // NOTE: Our current tuple order is [S, O, P]
 *   ]
 * }
 *
 * This controller normalizes each triplet row to:
 * { id, sentence, subject, predicate, object, include, deleted }
 */

import { upsertTripletsToKnowledgeGraphJob, getJobStatus } from "./pipeline_client.js";
import { showOversightOverlay, hideOversightOverlay } from "./oversight_overlay.js";
import { getOversightSource, getKeyword } from "./state/oversight_state.js";
import { setOversightStep, OversightSteps } from "./oversight_stepper.js";
import { showToast } from "./toast.js";
import { switchToTopLevelTab, TopLevelTabs } from "./utils/tabs.js"
import { refreshGraphForKeyword } from "./graph_refresh.js";
import { fireConfetti } from "./confetti.js";
import { resetPipelineSession } from "./ui_reset.js";
import { exportTripletReviewAsXlsx } from "./utils/export_utils.js";


/** Internal in-memory store of editable rows. */
const state = {
    rows: [],       // Array<TripletRow>
    skipped: [],    // Array<{sentence: string, reason: string}>
    allowedPredicates: [],
    filter: "",     // current filter string
    deletedCount: 0 // derived
};

// Getters
const getTableWrap = () => document.getElementById("triplets-table-wrap");
const getEmptyStateEl = () => document.getElementById("triplets-empty");
const getSkippedWrap = () => document.getElementById("triplets-skipped-wrap");
const getCountEl = () => document.getElementById("triplets-count");
const getDeletedCountEl = () => document.getElementById("triplets-deleted-count");
const getExcludedCountEl = () => document.getElementById("triplets-excluded-count");
const getFilterInput = () => document.getElementById("triplets-filter");
const getSubmitBtn = () => document.getElementById("triplets-submit");
const getBackBtn = () => document.getElementById("triplets-back");
const getClearFilterBtn = () => document.getElementById("triplets-clear-filter");
const getBottomEl = () => document.getElementById("oversight-bottom");

// Cached last extraction result (for navigation without re-calling API)
let _hasTripletsCache = false;

export function hasTripletsCache() {
    return _hasTripletsCache && (state.rows.length > 0 || state.skipped.length > 0);
}

/**
 * @typedef {Object} TripletRow
 * @property {string} id Unique stable row id.
 * @property {string} sentence Source sentence (from the extraction result).
 * @property {string} subject Editable subject.
 * @property {string} predicate Editable predicate.
 * @property {string} object Editable object.
 * @property {boolean} include Whether this row should be submitted to the KG.
 * @property {boolean} deleted Soft delete flag.
 */

/**
 * Generate a stable-ish id for a triplet row.
 * Uses sentence + SPO fields; good enough for UI row identity.
 */
function rowId({ sentence, subject, predicate, object }, idx) {
    return `${idx}::${sentence}::${subject}::${predicate}::${object}`;
}

/**
 * Given job.result.extracted_triplets, flatten to TripletRow[].
 * Our backend currently returns triplets in order: [subject, object, predicate].
 * We normalize to subject/predicate/object for editing UI.
 *
 * @param {any} extractedTripletsList job.result.extracted_triplets
 * @returns {TripletRow[]}
 */
function normalizeExtractionResult(extractedTripletsList) {
    const rows = [];
    const skipped = [];
    let idx = 0;

    (extractedTripletsList || []).forEach(item => {
        const sentence = (item?.sentence ?? "").trim();
        const triplets = Array.isArray(item?.triplets) ? item.triplets : [];
        const before = rows.length;
        const decision = String(item?.decision || "").trim().toUpperCase();
        const maxScore = Number.isFinite(Number(item?.max_score)) ? Number(item.max_score) : null;
        const matchedNeighborSentence = String(item?.matched_neighbor_sentence || "").trim();
        const upsertEligible = item?.upsert_eligible !== false;

        triplets.forEach(t => {
            const s = String(t?.[0] ?? "").trim();
            const o = String(t?.[1] ?? "").trim();
            const p = String(t?.[2] ?? "").trim();

            // Skip empty junk rows defensively
            if (!s || !p || !o) return;

            const row = {
                id: rowId({ sentence, subject: s, predicate: p, object: o }, idx++),
                sentence,
                originalQuality: String(item?.original_quality || sentence).trim(),
                sourceContext: item?.source_context ?? null,
                decision,
                maxScore,
                matchedNeighborSentence,
                upsertEligible,
                include: upsertEligible,
                subject: s,
                predicate: p,
                object: o,
                deleted: false,
            };

            rows.push(row);
        });

        if (rows.length === before) {
            const reason = String(item?.skipped_reason || "No allowed relationship type fit this quality.").trim();
            const originalQuality = String(item?.original_quality || sentence).trim();
            const sourceContext = item?.source_context ?? null;
            if (sentence || reason) {
                skipped.push({
                    sentence,
                    originalQuality,
                    sourceContext,
                    reason,
                    decision,
                    maxScore,
                    matchedNeighborSentence,
                    upsertEligible,
                });
            }
        }
    });

    return { rows, skipped };
}

/**
 * Public entrypoint:
 * Call this when triplet extraction job finishes (job.state === "done").
 *
 * @param {any} jobResult job.result (from GET /jobs/<id>)
 */
export function renderExtractedTripletsFromJobResult(jobResult, { activate = true } = {}) {
    if (activate) {
        setOversightStep(OversightSteps.EXTRACTED_TRIPLETS);
    }

    const extractedTriplets = jobResult?.extracted_triplets ?? [];
    const normalized = normalizeExtractionResult(extractedTriplets);

    state.rows = normalized.rows;
    state.skipped = normalized.skipped;
    state.allowedPredicates = Array.isArray(jobResult?.allowed_predicates)
        ? jobResult.allowed_predicates.map(String).filter(Boolean)
        : [];
    _hasTripletsCache = state.rows.length > 0 || state.skipped.length > 0; // cache rows or skipped notices for navigation without re-calling the API
    state.filter = "";

    // Render
    wireTripletsToolbar(); // idempotent
    renderTripletsTable();
    renderSkippedNotices();
    updateTripletsCounters();
}

/**
 * to "show existing table again" without needing jobResult. Useful when user navigates back and forth between steps.
 */
export function showCachedTripletsStep() {
    if (!hasTripletsCache()) return false;
    setOversightStep(OversightSteps.EXTRACTED_TRIPLETS);
    wireTripletsToolbar(); // idempotent
    renderTripletsTable();
    renderSkippedNotices();
    updateTripletsCounters();
    return true;
}

/**
 * Wire toolbar buttons + filter input once.
 * Safe to call multiple times (listeners are set with "once" guards).
 */
function wireTripletsToolbar() {
    const backBtn = getBackBtn();
    if (backBtn && !backBtn.dataset.wired) {
        backBtn.dataset.wired = "1";
        backBtn.addEventListener("click", () => {
            // go back to qualities selection UI
            setOversightStep(OversightSteps.CANDIDATE_SENTENCES);
        });
    }

    const filterInput = getFilterInput();
    if (filterInput && !filterInput.dataset.wired) {
        filterInput.dataset.wired = "1";
        filterInput.addEventListener("input", () => {
            state.filter = (filterInput.value ?? "").trim().toLowerCase();
            renderTripletsTable();
            updateTripletsCounters();
        });
    }

    const clearBtn = getClearFilterBtn();
    if (clearBtn && !clearBtn.dataset.wired) {
        clearBtn.dataset.wired = "1";
        clearBtn.addEventListener("click", () => {
            state.filter = "";
            const inp = getFilterInput();
            if (inp) inp.value = "";
            renderTripletsTable();
            updateTripletsCounters();
        });
    }

    const submitBtn = getSubmitBtn();
    if (submitBtn && !submitBtn.dataset.wired) {
        submitBtn.dataset.wired = "1";
        submitBtn.addEventListener("click", async () => {
            console.log("Submitting triplets to KG with payload:");
            await submitTripletsToKnowledgeGraph();
        });
    }

    // Export
    const exportBtn = document.getElementById("triplets-export");
    if (exportBtn && !exportBtn.dataset.wired) {
        exportBtn.dataset.wired = "1";
        exportBtn.addEventListener("click", () => {
            exportTripletsReviewAsXlsx();
        });
    }
}

function formatScore(score) {
    return Number.isFinite(score) ? score.toFixed(2) : "n/a";
}

function formatDecision(decision) {
    return String(decision || "UNKNOWN").replaceAll("_", " ");
}

function decisionBadgeClass(decision) {
    if (decision === "NEW") return "text-bg-success";
    if (decision === "PARTIALLY_NEW") return "text-bg-warning";
    if (decision === "EXISTING") return "text-bg-secondary";
    return "text-bg-light";
}

function tripletPopoverTitle(row) {
    const ctx = row?.sourceContext;
    const rawIndex = ctx?.source_doc_index;
    if (Number.isInteger(rawIndex)) {
        return `Original quality and source paragraph ${rawIndex + 1}`;
    }
    return "Original quality and source";
}

function provenanceText(item) {
    const parts = [];
    const quality = String(item?.originalQuality || item?.sentence || "").trim();
    if (quality) parts.push(`Original quality:\n${quality}`);

    const decision = String(item?.decision || "").trim();
    if (decision) parts.push(`Novelty decision: ${formatDecision(decision)}`);

    if (Number.isFinite(item?.maxScore)) {
        parts.push(`Similarity score: ${formatScore(item.maxScore)}`);
    }

    const neighbor = String(item?.matchedNeighborSentence || "").trim();
    if (neighbor) parts.push(`Nearest KG match:\n${neighbor}`);

    const ctx = item?.sourceContext;
    const metadata = ctx?.metadata || {};
    const headings = Array.isArray(metadata?.headings)
        ? metadata.headings.map(String).filter(Boolean)
        : [];
    const sourceText = String(ctx?.source_text || "").trim();

    if (headings.length > 0 || sourceText) {
        const sourceParts = [];
        if (headings.length > 0) sourceParts.push(headings.join(" > "));
        if (sourceText) sourceParts.push(sourceText);
        parts.push(`Source chunk:\n${sourceParts.join("\n\n")}`);
    }

    return parts.join("\n\n") || "No source context was attached to this item.";
}

function tripletPopoverContent(row) {
    return provenanceText(row);
}

function renderSkippedNotices() {
    const wrap = getSkippedWrap();
    if (!wrap) return;

    if (!state.skipped.length) {
        wrap.classList.add("d-none");
        wrap.innerHTML = "";
        return;
    }

    wrap.classList.remove("d-none");
    wrap.innerHTML = `
      <div class="fw-semibold mb-1">${state.skipped.length} qualities did not produce allowed triplets.</div>
      ${state.skipped.map(item => `
        <details class="triplets-skipped-item">
          <summary><span class="fw-semibold">${escapeHtml(item.reason)}</span> ${escapeHtml(item.originalQuality || item.sentence)}</summary>
          <pre class="mb-0 mt-1 small">${escapeHtml(provenanceText(item))}</pre>
        </details>
      `).join("")}
    `;
}

function isPredicateAllowed(predicate) {
    return state.allowedPredicates.length === 0 || state.allowedPredicates.includes(predicate);
}

/**
 * Apply filter + hide deleted rows? (We keep deleted visible but greyed, so user can undo later.)
 * Filtering checks subject/predicate/object/sentence.
 */
function getVisibleRows() {
    const q = state.filter;
    if (!q) return state.rows;

    return state.rows.filter(r => {
        // const hay = `${r.subject} ${r.predicate} ${r.object} ${r.sentence}`.toLowerCase();
        const hay = `${r.subject} ${r.predicate} ${r.object}`.toLowerCase();
        return hay.includes(q); // i.e., show the row if the filter query is a substring of any of the fields 
    });
}

/** Render the editable triplets table into #triplets-table-wrap. */
function renderTripletsTable() {
    const wrap = getTableWrap();
    const empty = getEmptyStateEl();
    if (!wrap) return;

    const rows = getVisibleRows();

    if (empty) empty.classList.toggle("d-none", rows.length !== 0);
    wrap.innerHTML = "";

    if (rows.length === 0) return;

    const table = document.createElement("table");
    table.className = "table table-hover align-middle";

    table.innerHTML = `
    <thead>
      <tr>
        <th style="width: 82px;">Include</th>
        <th style="width: 44px;"></th>
        <th>Subject</th>
        <th>Predicate</th>
        <th>Object</th>
        <th style="width: 132px;">Decision</th>
        <th style="width: 84px;" class="text-end">Actions</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;

    const tbody = table.querySelector("tbody");

    rows.forEach(r => {
        const tr = document.createElement("tr");
        if (r.deleted) tr.classList.add("opacity-50"); // soft-delete: visually indicate deleted rows but keep them visible

        tr.innerHTML = `
      <td class="text-center">
        <input class="form-check-input triplet-include" type="checkbox" ${r.include ? "checked" : ""} ${r.deleted ? "disabled" : ""} title="Include this triplet in KG submission" />
        <div class="small ${r.include ? "text-success" : "text-muted"}">${r.include ? "Included" : "Excluded"}</div>
      </td>

      <td>
        <button type="button"
          class="btn btn-sm btn-link p-0 text-muted"
          data-bs-toggle="popover"
          data-bs-trigger="click"
          data-bs-placement="right"
          data-bs-container="body"
          data-bs-custom-class="source-popover"
          data-bs-title="${escapeHtml(tripletPopoverTitle(r))}"
          data-bs-content="${escapeHtml(tripletPopoverContent(r))}">
          <i class="bi bi-chat-left-text"></i>
        </button>
      </td>

      <td>${editableCell("subject", r)}</td>
      <td>${editableCell("predicate", r)}</td>
      <td>${editableCell("object", r)}</td>
      <td><span class="badge ${decisionBadgeClass(r.decision)}">${escapeHtml(formatDecision(r.decision))}</span></td>

      <td class="text-end">
        <div class="btn-group btn-group-sm">
          <button type="button" class="btn btn-sm btn-outline-danger triplet-toggle-delete" title="Delete row">
            <i class="bi bi-trash"></i>
          </button>
        </div>
      </td>
    `;

        const includeCb = tr.querySelector(".triplet-include");
        if (includeCb) {
            includeCb.addEventListener("change", () => {
                r.include = includeCb.checked;
                renderTripletsTable();
                updateTripletsCounters();
            });
        }

        const btn = tr.querySelector(".triplet-toggle-delete");
        const icon = btn.querySelector("i");

        // Apply correct delete/undo appearance
        if (r.deleted) {
            btn.classList.add("btn-outline-secondary");
            btn.classList.remove("btn-outline-danger");
            btn.title = "Undo delete";
            icon.className = "bi bi-arrow-counterclockwise";
        } else {
            btn.classList.add("btn-outline-danger");
            btn.classList.remove("btn-outline-secondary");
            btn.title = "Delete row";
            icon.className = "bi bi-trash";
        }

        btn.addEventListener("click", () => {
            r.deleted = !r.deleted;
            renderTripletsTable();
            updateTripletsCounters();
        });

        // Wire inline edits
        wireEditableInputs(tr, r);
        tbody.appendChild(tr);
    });

    wrap.appendChild(table);

    initBootstrapPopovers(wrap);
}

/**
 * Render a Bootstrap-styled editable input for a field.
 * We use input-sm to keep the table compact.
 */
function editableCell(field, row) {
    const value = escapeHtml(row[field] ?? "");
    const disabled = row.deleted ? "disabled" : "";
    return `
    <input
      type="text"
      class="form-control form-control-sm"
      data-field="${field}"
      value="${value}"
      ${disabled}
    />
  `;
}

/**
 * After row HTML is inserted, attach listeners to inputs to update state.
 * @param {HTMLElement} tr
 * @param {TripletRow} row
 */
function wireEditableInputs(tr, row) {
    tr.querySelectorAll("input[data-field]").forEach(inp => {
        inp.addEventListener("input", () => {
            const field = inp.dataset.field;
            const v = (inp.value ?? "").trim();
            row[field] = v;
            updateTripletsCounters();
        });
    });
}

/**
 * Update counters + enable/disable KG submit button.
 * Rules:
 * - Only count rows that are NOT deleted AND have non-empty S/P/O.
 */
function updateTripletsCounters() {
    const countEl = getCountEl();
    const delEl = getDeletedCountEl();
    const excludedEl = getExcludedCountEl();
    const submitBtn = getSubmitBtn();

    const deleted = state.rows.filter(r => r.deleted).length;
    const excluded = state.rows.filter(r => !r.deleted && !r.include).length;

    const invalidPredicateCount = state.rows.filter(r => {
        if (r.deleted || !r.include || !r.predicate?.trim()) return false;
        return !isPredicateAllowed(r.predicate.trim());
    }).length;

    const valid = state.rows.filter(r => {
        if (r.deleted || !r.include) return false;
        return Boolean(r.subject?.trim() && r.predicate?.trim() && r.object?.trim() && isPredicateAllowed(r.predicate.trim()));
    }).length;

    if (countEl) countEl.textContent = String(valid);
    if (delEl) delEl.textContent = String(deleted);
    if (excludedEl) excludedEl.textContent = String(excluded);

    if (submitBtn) {
        submitBtn.disabled = (valid === 0 || invalidPredicateCount > 0);
        submitBtn.title = invalidPredicateCount > 0
            ? "One or more included predicates are not in the allowed relationship type list."
            : "";
    }
}

/**
 * Build the payload to send to the server for KG upsert.
 *
 * The Python upsert expects: Sequence[ExtractionResult], where each item is:
 *   { sentence: str, triplets: [(subject, object, predicate), ...] }
 *
 * We therefore:
 * - drop deleted rows
 * - drop rows with missing S/P/O
 * - group remaining rows by sentence
 * - emit `extractions` in the exact expected shape
 *
 * @returns {{ extractions: Array<{sentence: string, triplets: Array<[string,string,string]>}>, source?: string }}
 */
function buildUpsertPayload() {
    const rows = state.rows
        .filter(r => !r.deleted && r.include)
        .map(r => ({
            sentence: (r.sentence ?? "").trim(),
            subject: (r.subject ?? "").trim(),
            predicate: (r.predicate ?? "").trim(),
            object: (r.object ?? "").trim(),
        }))
        .filter(r => r.sentence && r.subject && r.predicate && r.object);

    /** @type {Map<string, Array<[string,string,string]>>} */
    const bySentence = new Map();

    for (const r of rows) {
        if (!bySentence.has(r.sentence)) bySentence.set(r.sentence, []);
        // IMPORTANT: backend expects (Subject, Object, Predicate)
        bySentence.get(r.sentence).push([r.subject, r.object, r.predicate]);
    }

    // Before submitting, we might want to deduplicate identical triplets inside each sentence (LLM sometimes repeats):
    for (const [sentence, triplets] of bySentence.entries()) {
        const seen = new Set();
        const uniq = [];
        for (const t of triplets) {
            const key = t.join("||");
            if (seen.has(key)) continue;
            seen.add(key);
            uniq.push(t);
        }
        bySentence.set(sentence, uniq);
    }

    // ✅ Convert Map -> expected payload array
    const extractions = Array.from(bySentence.entries())
        .map(([sentence, triplets]) => ({ sentence, triplets }))
        .filter(x => x.triplets.length > 0);

    const source = getOversightSource();

    return source ? { extractions, source } : { extractions };
}

/**
 * Submit edited triplets to server for KG upsert (Stage 7).
 * Uses the same job polling pattern as Stage 6.
 */
async function submitTripletsToKnowledgeGraph() {
    const payload = buildUpsertPayload();

    if (!payload.extractions.length) {
        showToast({
            type: "warning",
            title: "⚠️ Nothing to submit",
            message: "There are no valid triplets to insert into the Knowledge Graph.",
        });
        return;
    }

    showOversightOverlay(
        "Upserting triplets…",
        "Inserting reviewed triplets into the Knowledge Graph."
    );

    try {
        const start = await upsertTripletsToKnowledgeGraphJob(payload);
        const jobId = start.job_id;

        while (true) {
            const job = await getJobStatus(jobId);

            if (job.state === "done") {
                // Keep overlay up while we refresh, so user doesn't see stale graph.
                showOversightOverlay("🔄 Refreshing graph…", "Fetching updated subgraph from the Knowledge Graph.");

                const keyword = getKeyword();
                if (keyword) {
                    await refreshGraphForKeyword(keyword);
                }

                // Now switch after refresh so the user immediately sees the updated state.
                switchToTopLevelTab({ tab: TopLevelTabs.GRAPH });

                hideOversightOverlay();

                showToast({
                    type: "success",
                    title: "🎉 Success",
                    message: "Triplets were successfully inserted into the Knowledge Graph.",
                });

                fireConfetti();

                // resetPipelineSession();

                break;
            }

            if (job.state === "error") {
                throw new Error(job.error || "KG upsert failed.");
            }

            await sleep(1000);
        }
    } catch (e) {
        showToast({
            type: "error",
            title: "❌ Upsert Failed",
            message: e.message || String(e),
        });
    } finally {
        hideOversightOverlay();
    }
}

let popoverOutsideClickWired = false;

function initBootstrapPopovers(root) {
    if (!window.bootstrap) return;

    root.querySelectorAll('[data-bs-toggle="popover"]').forEach(el => {
        const existing = bootstrap.Popover.getInstance(el);
        if (existing) existing.dispose();
        new bootstrap.Popover(el, { trigger: "click" });
    });

    if (!popoverOutsideClickWired) {
        popoverOutsideClickWired = true;
        document.addEventListener("click", (event) => {
            if (event.target.closest('[data-bs-toggle="popover"], .popover')) return;
            document.querySelectorAll('[data-bs-toggle="popover"]').forEach(el => {
                bootstrap.Popover.getInstance(el)?.hide();
            });
        });
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

function sleep(ms) {
    return new Promise(res => setTimeout(res, ms));
}

// =========== Export ===========
function exportTripletsReviewAsXlsx() {
    const result = exportTripletReviewAsXlsx({
        rows: state.rows,
        documentName: getOversightSource(),
        keyword: getKeyword(),
    });

    if (!result.ok) {
        showToast({ type: "warning", title: "Nothing to export", message: result.reason });
        return;
    }

    showToast({
        type: "success",
        title: "Exported",
        message: `Triplet review workbook downloaded (${result.count} rows).`,
    });
}


/**
 * Reset extracted triplets UI+cache for a fresh run.
 */
export function resetExtractedTripletsUI() {
    state.rows = [];
    state.skipped = [];
    state.allowedPredicates = [];
    state.filter = "";
    _hasTripletsCache = false;

    const wrap = getTableWrap();
    if (wrap) wrap.innerHTML = "";

    const empty = getEmptyStateEl();
    if (empty) empty.classList.add("d-none");

    const skipped = getSkippedWrap();
    if (skipped) {
        skipped.classList.add("d-none");
        skipped.innerHTML = "";
    }

    const countEl = getCountEl();
    if (countEl) countEl.textContent = "0";

    const delEl = getDeletedCountEl();
    if (delEl) delEl.textContent = "0";

    const excludedEl = getExcludedCountEl();
    if (excludedEl) excludedEl.textContent = "0";

    const filterInput = getFilterInput();
    if (filterInput) filterInput.value = "";

    const submitBtn = getSubmitBtn();
    if (submitBtn) submitBtn.disabled = true;

    // Hide extracted triplets step by default
    const bottom = getBottomEl();
    if (bottom) bottom.classList.add("d-none");
}
