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
import { formatPredicateLabel } from "./utils/predicate_format.js";
import { runVerification } from "./verification_controller.js";
import { fetchJson } from "./api_client.js";


/** Internal in-memory store of editable rows. */
const state = {
    rows: [],       // Array<TripletRow>
    skipped: [],    // Array<{sentence: string, reason: string}>
    allowedPredicates: [],
    filter: "",     // current text filter string
    tagFilter: "",  // current tag filter key ("" = all tags)
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
const getTagFilterSelect = () => document.getElementById("triplets-tag-filter");
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
function normalizeExtractionResult(extractedTripletsList, allowedPredicates) {
    const rows = [];
    const skipped = [];
    const allowed = new Set((allowedPredicates || []).map(String).filter(Boolean));
    let idx = 0;

    (extractedTripletsList || []).forEach(item => {
        const sentence = (item?.sentence ?? "").trim();
        const triplets = Array.isArray(item?.triplets) ? item.triplets : [];
        const before = rows.length;
        const decision = String(item?.decision || "").trim().toUpperCase();
        const maxScore = Number.isFinite(Number(item?.max_score)) ? Number(item.max_score) : null;
        const matchedNeighborSentence = String(item?.matched_neighbor_sentence || "").trim();
        const schemaStatus = String(item?.schema_status || "").trim().toUpperCase();
        const schemaTemplate = String(item?.schema_template || "").trim();
        const groundingConfidence = Number.isFinite(Number(item?.grounding_confidence)) ? Number(item.grounding_confidence) : null;
        const matchedSchemaNodes = Array.isArray(item?.matched_schema_nodes) ? item.matched_schema_nodes.map(String).filter(Boolean) : [];
        const inferredNodeTypes = Array.isArray(item?.inferred_node_types) ? item.inferred_node_types.map(String).filter(Boolean) : [];
        const schemaNotes = Array.isArray(item?.schema_notes) ? item.schema_notes.map(String).filter(Boolean) : [];
        const schemaGrounding = item?.schema_grounding ?? null;
        const upsertEligible = item?.upsert_eligible !== false;
        const docName = String(item?.source_context?.doc_name || item?.source_context?.metadata?.source || "").trim();
        const modality = String(item?.modality || "").trim().toUpperCase();

        triplets.forEach(t => {
            const s = String(t?.[0] ?? "").trim();
            const o = String(t?.[1] ?? "").trim();
            const p = String(t?.[2] ?? "").trim();

            // Skip empty junk rows defensively
            if (!s || !p || !o) return;

            // Non-standard predicates stay visible but start excluded so the
            // reviewer has to deliberately opt them in.
            const nonStandard = allowed.size > 0 && !allowed.has(p);

            const row = {
                id: rowId({ sentence, subject: s, predicate: p, object: o }, idx++),
                sentence,
                originalQuality: String(item?.original_quality || sentence).trim(),
                sourceContext: item?.source_context ?? null,
                docName,
                modality,
                decision,
                maxScore,
                matchedNeighborSentence,
                schemaStatus,
                schemaTemplate,
                groundingConfidence,
                matchedSchemaNodes,
                inferredNodeTypes,
                schemaNotes,
                schemaGrounding,
                upsertEligible,
                include: upsertEligible && !nonStandard,
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
                    schemaStatus,
                    schemaTemplate,
                    groundingConfidence,
                    matchedSchemaNodes,
                    inferredNodeTypes,
                    schemaNotes,
                    schemaGrounding,
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
    state.allowedPredicates = Array.isArray(jobResult?.allowed_predicates)
        ? jobResult.allowed_predicates.map(String).filter(Boolean)
        : [];

    const normalized = normalizeExtractionResult(extractedTriplets, state.allowedPredicates);

    state.rows = normalized.rows;
    state.skipped = normalized.skipped;
    _hasTripletsCache = state.rows.length > 0 || state.skipped.length > 0; // cache rows or skipped notices for navigation without re-calling the API
    state.filter = "";
    state.tagFilter = "";

    // Render
    wireTripletsToolbar(); // idempotent
    renderTagFilterOptions();
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
    renderTagFilterOptions();
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
            state.tagFilter = "";
            const inp = getFilterInput();
            if (inp) inp.value = "";
            const tagSel = getTagFilterSelect();
            if (tagSel) tagSel.value = "";
            renderTripletsTable();
            updateTripletsCounters();
        });
    }

    const tagFilterSel = getTagFilterSelect();
    if (tagFilterSel && !tagFilterSel.dataset.wired) {
        tagFilterSel.dataset.wired = "1";
        tagFilterSel.addEventListener("change", () => {
            state.tagFilter = (tagFilterSel.value ?? "").trim();
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

    // Pre-merge "Check against KB" (sentence-level comparison)
    const checkBtn = document.getElementById("triplets-check-kb");
    if (checkBtn && !checkBtn.dataset.wired) {
        checkBtn.dataset.wired = "1";
        checkBtn.addEventListener("click", () => { void checkAgainstKb(); });
    }
}

/** Triples currently included (checked, not deleted), shaped for the preview API. */
function includedTriplesForPreview() {
    return state.rows
        .filter(r => r.include && !r.deleted)
        .map(r => ({
            subject: String(r.subject ?? "").trim(),
            predicate: String(r.predicate ?? "").trim(),
            object: String(r.object ?? "").trim(),
            sentence: String(r.sentence ?? "").trim(),
        }))
        .filter(t => t.subject && t.predicate && t.object);
}

const PREVIEW_GROUPS = [
    { key: "CONFLICT", label: "Conflicts", cls: "kb-cat-conflict", icon: "bi-exclamation-octagon-fill",
      blurb: "This document contradicts the KB." },
    { key: "TENSION", label: "Tensions", cls: "kb-cat-tension", icon: "bi-exclamation-triangle-fill",
      blurb: "Same statement, different obligation strength." },
    { key: "RELATED", label: "Related", cls: "kb-cat-related", icon: "bi-link-45deg",
      blurb: "Same concepts, related differently in the KB." },
    { key: "NEW", label: "New", cls: "kb-cat-new", icon: "bi-stars",
      blurb: "Not in the KB yet — would be added." },
    { key: "EXISTING", label: "Already in KB", cls: "kb-cat-existing", icon: "bi-check-circle",
      blurb: "Duplicate of knowledge already in the graph." },
];

/** Compare the included triplets against the KB and render the result panel. */
async function checkAgainstKb() {
    const wrap = document.getElementById("triplets-preview-wrap");
    if (!wrap) return;

    const triples = includedTriplesForPreview();
    if (!triples.length) {
        showToast({ type: "warning", title: "Nothing to check",
            message: "Include at least one triplet first." });
        return;
    }

    wrap.classList.remove("d-none");
    wrap.innerHTML =
        '<div class="text-muted small d-flex align-items-center gap-2 p-2">' +
        '<span class="spinner-border spinner-border-sm"></span>' +
        `Comparing ${triples.length} triplet(s) against the knowledge graph…</div>`;

    try {
        const report = await fetchJson("/api/comparison/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ triples, source: getOversightSource() || "" }),
        });
        renderPreviewReport(report);
    } catch (e) {
        wrap.innerHTML = `<div class="alert alert-danger py-2 small mb-0">Check failed: ${escapeHtml(e.message || String(e))}</div>`;
    }
}

function renderPreviewReport(report) {
    const wrap = document.getElementById("triplets-preview-wrap");
    if (!wrap) return;

    const items = Array.isArray(report?.items) ? report.items : [];
    const s = report?.summary || {};

    if (!items.length) {
        wrap.innerHTML = `<div class="text-muted small p-2">No triplets to compare.</div>`;
        return;
    }

    const chip = (g) => {
        const n = s[g.key.toLowerCase()] || 0;
        return `<span class="kb-chip ${g.cls}" title="${escapeHtml(g.blurb)}">
                  <i class="bi ${g.icon}"></i> ${n} ${escapeHtml(g.label)}</span>`;
    };

    const groupsHtml = PREVIEW_GROUPS.map(g => {
        const rows = items.filter(it => it.category === g.key);
        if (!rows.length) return "";
        const body = rows.map(it => {
            const kb = (it.kb_matches || []).map(m => `
              <div class="kb-side kb-side-graph">
                <span class="kb-side-tag">KB${m.docs && m.docs.length ? " · " + escapeHtml(m.docs.join(", ")) : ""}</span>
                <span class="kb-pred">${escapeHtml(m.predicate)}${m.modality ? " · " + escapeHtml(m.modality) : ""}</span>
                <div class="kb-sentence">${escapeHtml(m.sentence || "—")}</div>
              </div>`).join("");
            return `
              <div class="kb-item ${g.cls}">
                <div class="kb-triple">
                  <code>${escapeHtml(it.subject)}</code>
                  <span class="kb-pred">${escapeHtml(it.predicate)}${it.modality ? " · " + escapeHtml(it.modality) : ""}</span>
                  <code>${escapeHtml(it.object)}</code>
                </div>
                <div class="kb-reason">${escapeHtml(it.reason || "")}</div>
                <div class="kb-side kb-side-doc">
                  <span class="kb-side-tag">Your doc</span>
                  <div class="kb-sentence">${escapeHtml(it.sentence || "—")}</div>
                </div>
                ${kb}
              </div>`;
        }).join("");
        return `<div class="kb-group">
                  <div class="kb-group-head ${g.cls}"><i class="bi ${g.icon}"></i> ${escapeHtml(g.label)} (${rows.length})</div>
                  ${body}
                </div>`;
    }).join("");

    wrap.innerHTML = `
      <div class="kb-preview-head d-flex align-items-center flex-wrap gap-2 mb-2">
        <span class="fw-semibold small"><i class="bi bi-shuffle me-1"></i>Compared ${s.total || items.length} triplet(s) against the KB (before merge):</span>
        ${PREVIEW_GROUPS.map(chip).join("")}
        <button type="button" class="btn btn-sm btn-link p-0 ms-auto" id="kb-preview-close">Hide</button>
      </div>
      ${groupsHtml}
    `;
    document.getElementById("kb-preview-close")?.addEventListener("click", () => {
        wrap.classList.add("d-none");
    });
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

function schemaStatusLabel(status) {
    const value = String(status || "").toUpperCase();
    if (value === "SCHEMA_VALID") return "Schema-valid";
    if (value === "NEEDS_SCHEMA_REVIEW") return "Needs schema review";
    if (value === "NO_SCHEMA_FIT") return "No schema fit";
    return "Not checked";
}

function schemaRowClass(status) {
    const value = String(status || "").toUpperCase();
    if (value === "NEEDS_SCHEMA_REVIEW") return "table-warning";
    if (value === "NO_SCHEMA_FIT") return "table-light";
    return "";
}

/**
 * Whether the review session spans more than one source document.
 * Used to decide if rows should carry a document tag.
 */
function hasMultipleDocs() {
    const names = new Set(
        state.rows.map(r => String(r?.docName || "").trim()).filter(Boolean)
    );
    return names.size > 1;
}

/**
 * Build the full tag list for a row. All row signals (novelty decision,
 * schema status, non-standard predicate, source document) live in ONE
 * Tags column. Schema-valid rows intentionally get no schema tag.
 *
 * @returns {Array<{key:string,label:string,cls:string}>}
 */
function rowTags(r, { multiDoc = hasMultipleDocs() } = {}) {
    const tags = [];

    const decision = String(r?.decision || "").trim().toUpperCase();
    if (decision) {
        tags.push({
            key: `decision:${decision}`,
            label: formatDecision(decision),
            cls: decisionBadgeClass(decision),
        });
    }

    const status = String(r?.schemaStatus || "").trim().toUpperCase();
    if (status === "NEEDS_SCHEMA_REVIEW") {
        tags.push({ key: "schema:NEEDS_SCHEMA_REVIEW", label: "Needs schema review", cls: "text-bg-warning" });
    } else if (status === "NO_SCHEMA_FIT") {
        tags.push({ key: "schema:NO_SCHEMA_FIT", label: "No schema fit", cls: "text-bg-danger" });
    }
    // SCHEMA_VALID and "not checked" intentionally produce no tag.

    const predicate = String(r?.predicate || "").trim();
    if (predicate && !isPredicateAllowed(predicate)) {
        tags.push({
            key: "predicate:NON_STANDARD",
            label: "Non-standard predicate",
            cls: "text-bg-danger",
        });
    }

    const modality = String(r?.modality || "").trim().toUpperCase();
    if (modality && modality !== "NONE") {
        const modalityCls = modality === "MANDATORY" ? "text-bg-primary"
            : modality === "PROHIBITED" ? "text-bg-dark"
            : "text-bg-secondary";
        tags.push({ key: `modality:${modality}`, label: modality.toLowerCase(), cls: modalityCls });
    }

    const docName = String(r?.docName || "").trim();
    if (docName && multiDoc) {
        tags.push({ key: `doc:${docName}`, label: docName, cls: "text-bg-info" });
    }

    return tags;
}

/**
 * Populate the tag filter dropdown from the tags present in current rows.
 * Keeps the current selection when still available.
 */
function renderTagFilterOptions() {
    const sel = getTagFilterSelect();
    if (!sel) return;

    const multiDoc = hasMultipleDocs();
    const seen = new Map(); // key -> label
    for (const r of state.rows) {
        for (const tag of rowTags(r, { multiDoc })) {
            if (!seen.has(tag.key)) seen.set(tag.key, tag.label);
        }
    }

    const current = state.tagFilter;
    const options = ['<option value="">All tags</option>'];
    for (const [key, label] of seen.entries()) {
        const selected = key === current ? "selected" : "";
        options.push(`<option value="${escapeHtml(key)}" ${selected}>${escapeHtml(label)}</option>`);
    }
    sel.innerHTML = options.join("");

    if (current && !seen.has(current)) {
        state.tagFilter = "";
        sel.value = "";
    }
}

function formatGroundingConfidence(value) {
    return Number.isFinite(value) ? value.toFixed(2) : "n/a";
}

function formatSchemaTemplateText(value) {
    return String(value || "").replace(/--([A-Za-z_][A-Za-z0-9_]*)-->/g, (_match, predicate) => {
        return `--${formatPredicateLabel(predicate)}-->`;
    });
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

    const docName = String(item?.docName || item?.sourceContext?.doc_name || item?.sourceContext?.metadata?.source || "").trim();
    if (docName) parts.push(`Document: ${docName}`);

    const decision = String(item?.decision || "").trim();
    if (decision) parts.push(`Novelty decision: ${formatDecision(decision)}`);

    if (Number.isFinite(item?.maxScore)) {
        parts.push(`Similarity score: ${formatScore(item.maxScore)}`);
    }

    const neighbor = String(item?.matchedNeighborSentence || "").trim();
    if (neighbor) parts.push(`Nearest KG match:\n${neighbor}`);

    const schemaStatus = String(item?.schemaStatus || "").trim();
    const schemaParts = [];
    if (schemaStatus) schemaParts.push(`Status: ${schemaStatusLabel(schemaStatus)}`);
    if (Number.isFinite(item?.groundingConfidence)) {
        schemaParts.push(`Grounding confidence: ${formatGroundingConfidence(item.groundingConfidence)}`);
    }
    if (Array.isArray(item?.matchedSchemaNodes) && item.matchedSchemaNodes.length) {
        schemaParts.push(`Schema hints: ${item.matchedSchemaNodes.join(", ")}`);
    }
    if (Array.isArray(item?.inferredNodeTypes) && item.inferredNodeTypes.length) {
        schemaParts.push(`Inferred node types: ${item.inferredNodeTypes.join(", ")}`);
    }
    if (item?.schemaTemplate) schemaParts.push(`Schema template:\n${item.schemaTemplate}`);
    if (Array.isArray(item?.schemaNotes) && item.schemaNotes.length) {
        schemaParts.push(`Schema notes:\n${item.schemaNotes.join("\n")}`);
    }
    if (schemaParts.length) parts.push(`Schema grounding:\n${schemaParts.join("\n\n")}`);

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
    if (state.allowedPredicates.length === 0) return true;
    const needle = String(predicate || "").trim().toLowerCase();
    return state.allowedPredicates.some(p => String(p).toLowerCase() === needle);
}

/**
 * Apply text + tag filters. Deleted rows stay visible but greyed, so user can undo later.
 * Text filtering checks subject/predicate/object; tag filtering checks row tags.
 */
function getVisibleRows() {
    const q = state.filter;
    const tag = state.tagFilter;
    const multiDoc = hasMultipleDocs();

    return state.rows.filter(r => {
        if (tag) {
            const keys = rowTags(r, { multiDoc }).map(t => t.key);
            if (!keys.includes(tag)) return false;
        }
        if (!q) return true;
        const hay = `${r.subject} ${r.predicate} ${formatPredicateLabel(r.predicate)} ${r.object}`.toLowerCase();
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
        <th style="width: 92px;">
          <div class="d-flex align-items-center gap-1">
            <input class="form-check-input triplet-include-all" type="checkbox"
                   title="Include / exclude all shown triplets" aria-label="Include all" />
            <span>Include</span>
          </div>
        </th>
        <th style="width: 44px;"></th>
        <th>Subject</th>
        <th>Predicate</th>
        <th>Object</th>
        <th style="width: 220px;">Tags</th>
        <th style="width: 84px;" class="text-end">Actions</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;

    const tbody = table.querySelector("tbody");
    const multiDoc = hasMultipleDocs();

    rows.forEach(r => {
        const tr = document.createElement("tr");
        const tags = rowTags(r, { multiDoc });
        const schemaClass = schemaRowClass(r.schemaStatus);
        if (schemaClass) tr.classList.add(schemaClass);
        else if (tags.some(t => t.key === "predicate:NON_STANDARD")) tr.classList.add("table-warning");
        if (r.deleted) tr.classList.add("opacity-50"); // soft-delete: visually indicate deleted rows but keep them visible

        const tagsHtml = tags.length
            ? tags.map(t => `<span class="badge ${t.cls} me-1 mb-1">${escapeHtml(t.label)}</span>`).join("")
            : `<span class="text-muted small">—</span>`;

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
      <td>${tagsHtml}</td>

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

    // Header "include all" checkbox: toggles every shown, non-deleted row.
    const selectAll = table.querySelector(".triplet-include-all");
    if (selectAll) {
        const toggleable = rows.filter(r => !r.deleted);
        const includedCount = toggleable.filter(r => r.include).length;
        selectAll.checked = toggleable.length > 0 && includedCount === toggleable.length;
        selectAll.indeterminate = includedCount > 0 && includedCount < toggleable.length;
        selectAll.disabled = toggleable.length === 0;
        selectAll.addEventListener("change", () => {
            const visible = getVisibleRows();
            visible.forEach(r => { if (!r.deleted) r.include = selectAll.checked; });
            renderTripletsTable();
            updateTripletsCounters();
        });
    }

    // Wrap in a horizontally-scrollable container so the Actions/delete column
    // is never clipped when the settings sidebar narrows the main area.
    const responsive = document.createElement("div");
    responsive.className = "table-responsive";
    responsive.appendChild(table);
    wrap.appendChild(responsive);

    initBootstrapPopovers(wrap);
}

/**
 * Render a Bootstrap-styled editable input for a field.
 * We use input-sm to keep the table compact.
 */
function editableCell(field, row) {
    if (field === "predicate") return predicateCell(row);

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

function predicateCell(row) {
    // Free-text box (same as subject/object). The reviewer can type any
    // predicate; off-vocabulary ones are flagged via the "Non-standard
    // predicate" tag rather than being constrained by a dropdown.
    const rawPredicate = String(row?.predicate || "").trim();
    const disabled = row.deleted ? "disabled" : "";
    return `
    <input
      type="text"
      class="form-control form-control-sm"
      data-field="predicate"
      value="${escapeHtml(rawPredicate)}"
      title="${escapeHtml(formatPredicateLabel(rawPredicate))}"
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
    tr.querySelectorAll("input[data-field], select[data-field]").forEach(inp => {
        inp.addEventListener("input", () => {
            const field = inp.dataset.field;
            const v = (inp.value ?? "").trim();
            row[field] = v;
            updateTripletsCounters();

            // Predicate edits can add/remove the "Non-standard predicate" tag,
            // so refresh the table and the tag filter options.
            if (field === "predicate") {
                renderTagFilterOptions();
                renderTripletsTable();
            }
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

    // Non-standard predicates are allowed in submission once the reviewer has
    // explicitly included the row — they are tagged, not blocked.
    const valid = state.rows.filter(r => {
        if (r.deleted || !r.include) return false;
        return Boolean(r.subject?.trim() && r.predicate?.trim() && r.object?.trim());
    }).length;

    const includedNonStandard = state.rows.filter(r => {
        if (r.deleted || !r.include || !r.predicate?.trim()) return false;
        return !isPredicateAllowed(r.predicate.trim());
    }).length;

    if (countEl) countEl.textContent = String(valid);
    if (delEl) delEl.textContent = String(deleted);
    if (excludedEl) excludedEl.textContent = String(excluded);

    if (submitBtn) {
        submitBtn.disabled = (valid === 0);
        submitBtn.title = includedNonStandard > 0
            ? `${includedNonStandard} included triplet(s) use a non-standard predicate. They will be written with a sanitized relationship type.`
            : "";
    }
}

/**
 * Build the payload to send to the server for KG upsert.
 *
 * The Python upsert expects: Sequence[ExtractionResult], where each item is:
 *   { sentence: str, triplets: [(subject, object, predicate), ...], provenance?: {...} }
 *
 * We therefore:
 * - drop deleted rows
 * - drop rows with missing S/P/O
 * - group remaining rows by (document, sentence) so provenance stays per-doc
 * - attach {doc_name, quality, chunk_index, chunk_excerpt} provenance per group
 * - emit `extractions` in the exact expected shape
 *
 * @returns {{ extractions: Array<{sentence: string, triplets: Array<[string,string,string]>, provenance?: Object}>, source?: string }}
 */
function buildUpsertPayload() {
    const fallbackDoc = getOversightSource() || "";

    const rows = state.rows
        .filter(r => !r.deleted && r.include)
        .map(r => ({
            sentence: (r.sentence ?? "").trim(),
            subject: (r.subject ?? "").trim(),
            predicate: (r.predicate ?? "").trim(),
            object: (r.object ?? "").trim(),
            row: r,
        }))
        .filter(r => r.sentence && r.subject && r.predicate && r.object);

    /** @type {Map<string, {sentence: string, triplets: Array<[string,string,string]>, provenance: Object|null}>} */
    const byGroup = new Map();

    for (const r of rows) {
        const provenance = rowProvenance(r.row, fallbackDoc);
        const groupKey = `${provenance?.doc_name || ""}::${r.sentence}`;

        if (!byGroup.has(groupKey)) {
            byGroup.set(groupKey, { sentence: r.sentence, triplets: [], provenance });
        }
        // IMPORTANT: backend expects (Subject, Object, Predicate)
        byGroup.get(groupKey).triplets.push([r.subject, r.object, r.predicate]);
    }

    // Deduplicate identical triplets inside each group (LLM sometimes repeats):
    for (const group of byGroup.values()) {
        const seen = new Set();
        const uniq = [];
        for (const t of group.triplets) {
            const key = t.join("||");
            if (seen.has(key)) continue;
            seen.add(key);
            uniq.push(t);
        }
        group.triplets = uniq;
    }

    // ✅ Convert Map -> expected payload array
    const extractions = Array.from(byGroup.values())
        .filter(x => x.triplets.length > 0)
        .map(({ sentence, triplets, provenance }) => (
            provenance ? { sentence, triplets, provenance } : { sentence, triplets }
        ));

    const source = getOversightSource();

    return source ? { extractions, source } : { extractions };
}

/**
 * Build the structured provenance payload for one row:
 * which document, which quality, which source chunk.
 */
function rowProvenance(row, fallbackDoc) {
    const ctx = row?.sourceContext || {};
    const docName = String(row?.docName || ctx?.doc_name || ctx?.metadata?.source || fallbackDoc || "").trim();
    const quality = String(row?.originalQuality || row?.sentence || "").trim();
    const rawIndex = ctx?.source_doc_index;
    const chunkIndex = Number.isInteger(rawIndex) ? rawIndex : null;
    const chunkExcerpt = String(ctx?.source_text || "").trim().slice(0, 500);
    const modality = String(row?.modality || "").trim().toUpperCase();

    if (!docName && !quality && chunkIndex === null && !chunkExcerpt) return null;

    const provenance = {
        doc_name: docName,
        quality,
        chunk_index: chunkIndex,
        chunk_excerpt: chunkExcerpt,
    };
    if (modality && modality !== "NONE") provenance.modality = modality;
    return provenance;
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

                // Phase B: verify the freshly-built graph. Non-blocking — the
                // graph is already upserted and shown; the verification panel
                // renders its own PASS/FAIL verdict on the graph view and any
                // error stays inside the panel.
                const verifySentences = payload.extractions
                    .map(e => e.sentence)
                    .filter(Boolean);
                runVerification({
                    sentences: verifySentences,
                    allowedPredicates: state.allowedPredicates || [],
                });

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
    state.tagFilter = "";
    _hasTripletsCache = false;

    const tagSel = getTagFilterSelect();
    if (tagSel) tagSel.innerHTML = '<option value="">All tags</option>';

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
