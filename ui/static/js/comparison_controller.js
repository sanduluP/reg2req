/**
 * Compare tab controller
 * ----------------------
 * Cross-document analysis views over the KG provenance layer:
 * - Source selector: pick which provenance sources to analyse
 * - Overlap & Coverage: per-document contribution, multi-doc assertions, concept matrix
 * - Alignment: SAME_AS candidate review (accept/reject)
 * - Conflicts: typed candidates + LLM verdicts, accept/dismiss into (:Conflict) nodes
 * - Ambiguity: undefined normative terms, vague language, rejected near-synonyms
 *
 * Heavy scans run as background jobs; this controller polls the shared
 * pipeline jobs endpoint.
 */

import {
    fetchComparisonSources,
    fetchOverlapReport,
    startAlignmentScan,
    postAlignmentDecision,
    startConflictScan,
    postConflictDecision,
    fetchRecordedConflicts,
    fetchAmbiguityReport,
} from "./comparison_client.js";
import { getRunContext } from "./state/oversight_state.js";
import { getJobStatus } from "./pipeline_client.js";
import { showToast } from "./toast.js";
import { exportRowsAsXlsx } from "./utils/export_utils.js";
import { formatPredicateLabel } from "./utils/predicate_format.js";

const state = {
    overlap: null,        // last overlap report
    alignment: [],        // pending alignment candidates
    conflicts: [],        // current conflict candidates
    ambiguity: null,      // last ambiguity report
    overlapLoaded: false,

    // Source selector
    allSources: [],       // [{name, type}]  type = "graph" | "session"
    selectedSources: null, // null = all; Set<string> = subset
};

/* ------------------------------ helpers ------------------------------ */

function el(id) {
    return document.getElementById(id);
}

function escapeHtml(str) {
    return String(str ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function setStatus(id, message) {
    const node = el(id);
    if (node) node.textContent = message || "";
}

function sleep(ms) {
    return new Promise(res => setTimeout(res, ms));
}

async function pollJob(jobId, { statusId, label }) {
    while (true) {
        const job = await getJobStatus(jobId);
        if (job.state === "done") return job.result;
        if (job.state === "error") throw new Error(job.error || `${label} failed.`);
        setStatus(statusId, job.message || `${label} running...`);
        await sleep(1500);
    }
}

function simpleTable(headers, rowsHtml, { emptyText = "Nothing to show yet." } = {}) {
    if (!rowsHtml.length) {
        return `<div class="text-muted small">${escapeHtml(emptyText)}</div>`;
    }
    return `
      <div class="table-responsive">
        <table class="table table-sm table-hover align-middle">
          <thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
          <tbody>${rowsHtml.join("")}</tbody>
        </table>
      </div>
    `;
}

function verdictBadgeClass(verdict) {
    const v = String(verdict || "").toUpperCase();
    if (v === "CONTRADICT") return "text-bg-danger";
    if (v === "TENSION") return "text-bg-warning";
    if (v === "AGREE") return "text-bg-success";
    if (v === "UNRELATED") return "text-bg-secondary";
    return "text-bg-light";
}

function conflictTypeLabel(type) {
    const t = String(type || "").toUpperCase();
    if (t === "MODALITY_CONFLICT") return "Modality";
    if (t === "DEFINITION_DIVERGENCE") return "Definition";
    if (t === "TAXONOMY_CONFLICT") return "Taxonomy";
    if (t === "VALUE_CONFLICT") return "Value";
    return t || "Unknown";
}

/**
 * Return the active source filter as an array, or null for "all".
 */
function activeSources() {
    if (!state.selectedSources) return null;
    const arr = [...state.selectedSources];
    return arr.length === 0 ? [] : arr;
}

/** How many documents are actually persisted in the graph (analysis works on these). */
function graphSourceCount() {
    return state.allSources.filter(s => s.type === "graph").length;
}

/** True if the current session has an uploaded doc not yet submitted to the KG. */
function hasUnsubmittedSessionDoc() {
    return state.allSources.some(s => s.type === "session");
}

/**
 * Cross-document analysis (alignment / conflicts / overlap) runs over documents
 * stored in Neo4j. If there are fewer than two, tell the reviewer exactly what
 * to do — usually "submit your uploaded document to the KG first".
 */
function notEnoughGraphDocsHint(kind) {
    if (graphSourceCount() >= 2) return "";
    if (hasUnsubmittedSessionDoc()) {
        return `Your uploaded document isn't in the knowledge graph yet, so there's nothing to ${kind} it against. ` +
            `Go to Review & extract → Submit to KG, then scan here.`;
    }
    return `Need at least two documents in the graph to ${kind}. Initialize the baseline graph and/or submit a document first.`;
}

/* ======================== Source selector ======================== */

async function loadSources() {
    const spinner = el("compare-sources-loading");
    if (spinner) spinner.classList.remove("d-none");

    // The Compare tab is a KB OVERVIEW: it only analyses documents already in
    // the knowledge graph (Neo4j). Pre-merge comparison happens elsewhere
    // ("Check against KB" on the review screen), so there is no session section.
    let neo4jSources = [];
    try {
        const data = await fetchComparisonSources();
        neo4jSources = Array.isArray(data.neo4j_sources) ? data.neo4j_sources : [];
    } catch (e) {
        console.warn("Could not fetch KB sources:", e);
    }

    state.allSources = neo4jSources.map(n => ({ name: n, type: "graph" }));
    state.selectedSources = null; // default: all selected

    _renderDocChecklist(neo4jSources);
    _updateDocsLabel();

    if (spinner) spinner.classList.add("d-none");
}

/** Render the KB documents as a checkbox list inside the selector dropdown. */
function _renderDocChecklist(sources) {
    const wrap = el("compare-graph-sources-wrap");
    const hint = el("compare-graph-sources-hint");
    if (!wrap) return;

    if (!sources.length) {
        wrap.innerHTML = `<div class="text-muted small px-1 py-2">No documents in the knowledge graph yet.</div>`;
        if (hint) hint.innerHTML =
            `Run the pipeline on a PDF, then <em>Submit to KG</em> — or use <em>Initialize graph</em> to load baseline knowledge.`;
        return;
    }
    if (hint) hint.textContent = "Tick documents to include in the analysis.";

    wrap.innerHTML = "";
    for (const name of sources) {
        const isSel = state.selectedSources === null || state.selectedSources.has(name);
        const item = document.createElement("label");
        item.className = "dropdown-item d-flex align-items-center gap-2 px-2 py-1 compare-doc-item";
        item.style.cssText = "cursor:pointer; font-size:0.78rem; border-radius:0.3rem;";
        item.innerHTML =
            `<input type="checkbox" class="form-check-input m-0 compare-doc-cb" ${isSel ? "checked" : ""}>` +
            `<span class="text-truncate" style="max-width:220px;" title="${escapeHtml(name)}">${escapeHtml(name)}</span>`;
        const cb = item.querySelector(".compare-doc-cb");
        cb.dataset.sourceName = name;
        cb.addEventListener("change", () => {
            if (state.selectedSources === null) {
                state.selectedSources = new Set(state.allSources.map(s => s.name));
            }
            if (cb.checked) state.selectedSources.add(name);
            else state.selectedSources.delete(name);
            if (state.selectedSources.size === state.allSources.length) state.selectedSources = null;
            _updateDocsLabel();
        });
        wrap.appendChild(item);
    }
}

/** Summarise the selection on the dropdown button. */
function _updateDocsLabel() {
    const label = el("compare-docs-label");
    if (!label) return;
    const total = state.allSources.length;
    if (total === 0) { label.textContent = "Documents"; return; }
    const sel = state.selectedSources === null ? total : state.selectedSources.size;
    label.textContent = (sel === total) ? `All documents (${total})` : `Documents (${sel}/${total})`;
}

function _updateDocCheckboxes() {
    document.querySelectorAll(".compare-doc-cb").forEach(cb => {
        cb.checked = state.selectedSources === null || state.selectedSources.has(cb.dataset.sourceName);
    });
}

function selectAllSources() {
    state.selectedSources = null;
    _updateDocCheckboxes();
    _updateDocsLabel();
}

function clearAllSources() {
    state.selectedSources = new Set();
    _updateDocCheckboxes();
    _updateDocsLabel();
}

/* ------------------------------ Overlap ------------------------------ */

async function refreshOverlap() {
    setStatus("compare-overlap-status", "Loading overlap report...");
    const sources = activeSources();
    try {
        state.overlap = await fetchOverlapReport({ sources });
        state.overlapLoaded = true;
        renderOverlap();
        const edgeCount = state.overlap?.num_edges_with_provenance ?? 0;
        const filterNote = sources ? ` (filtered to ${sources.length} source(s))` : "";
        setStatus("compare-overlap-status", `${edgeCount} edges with provenance${filterNote}.`);
    } catch (e) {
        setStatus("compare-overlap-status", `Failed: ${e.message || e}`);
    }
}

function renderOverlap() {
    const report = state.overlap || {};

    // 1) Document coverage — one stat card per document.
    const coverage = Array.isArray(report.coverage) ? report.coverage : [];
    const cards = coverage.map(c => `
      <div class="kb-stat-card">
        <div class="kb-stat-doc" title="${escapeHtml(c.doc)}"><i class="bi bi-file-earmark-text"></i>${escapeHtml(c.doc)}</div>
        <div class="kb-stat-row">
          <div class="kb-stat"><span class="kb-stat-num">${c.assertions}</span><span class="kb-stat-label">assertions</span></div>
          <div class="kb-stat"><span class="kb-stat-num">${c.concepts}</span><span class="kb-stat-label">concepts</span></div>
          <div class="kb-stat"><span class="kb-stat-num">${c.normative_statements}</span><span class="kb-stat-label">normative</span></div>
        </div>
      </div>`).join("");
    el("compare-coverage-wrap").innerHTML = `
      <div class="kb-section">
        <div class="kb-section-title"><i class="bi bi-files"></i>Document coverage</div>
        ${coverage.length
          ? `<div class="kb-stat-cards">${cards}</div>`
          : `<div class="kb-empty-card">No documents in the knowledge graph yet — run the pipeline, then <em>Submit to KG</em>.</div>`}
      </div>`;

    // 2) Assertions supported by 2+ documents (agreement / tension).
    const overlap = Array.isArray(report.overlap) ? report.overlap : [];
    const ovRows = overlap.map(o => {
        const verdict = String(o.verdict || "AGREEMENT").toUpperCase();
        const verdictBadge = verdict === "TENSION"
            ? `<span class="badge kb-verdict-tension" title="Same assertion, different obligation strength">⚠ Tension</span>`
            : `<span class="badge kb-verdict-agree" title="Documents agree">✓ Agreement</span>`;
        const docsCol = (o.docs || []).map(d => {
            const m = o.modality_by_doc?.[d];
            const mTag = m ? ` <span class="badge text-bg-light border" style="font-size:0.6rem;">${escapeHtml(m.toLowerCase())}</span>` : "";
            return `<span class="badge text-bg-secondary me-1" style="font-weight:500;">${escapeHtml(d)}${mTag}</span>`;
        }).join("");
        return `<tr>
          <td class="fw-semibold">${escapeHtml(o.source)}</td>
          <td class="text-muted">${escapeHtml(formatPredicateLabel(o.predicate))}</td>
          <td>${escapeHtml(o.target)}</td>
          <td>${verdictBadge}</td>
          <td>${docsCol}</td>
        </tr>`;
    });
    el("compare-overlap-wrap").innerHTML = `
      <div class="kb-section">
        <div class="kb-section-title"><i class="bi bi-intersect"></i>Shared assertions
          <span class="kb-section-sub">— same statement in 2+ documents · tension = different obligation strength</span>
        </div>
        ${overlap.length
          ? simpleTable(["Subject", "Predicate", "Object", "Verdict", "Documents"], ovRows)
          : `<div class="kb-empty-card">No assertion is supported by more than one document yet.</div>`}
      </div>`;

    // 3) Concept × document matrix — shared concepts marked.
    const concepts = report.concepts || {};
    const docs = Array.isArray(concepts.documents) ? concepts.documents : [];
    const rows = Array.isArray(concepts.rows) ? concepts.rows : [];
    const conceptRows = rows.slice(0, 100).map(r => {
        const shared = (r.docs || 0) > 1;
        const sharedBadge = shared ? `<span class="kb-badge-shared">shared</span>` : "";
        return `<tr class="${shared ? "kb-shared" : ""}">
          <td class="fw-semibold">${escapeHtml(r.concept)}${sharedBadge}</td>
          ${docs.map(d => { const n = r.counts?.[d] || 0; return `<td class="${n ? "" : "kb-zero"}">${n || "·"}</td>`; }).join("")}
          <td class="text-center">${r.docs}</td>
        </tr>`;
    }).join("");
    const header = `<th>Concept</th>${docs.map(d => `<th>${escapeHtml(d)}</th>`).join("")}<th class="text-center"># Docs</th>`;
    el("compare-concepts-wrap").innerHTML = `
      <div class="kb-section">
        <div class="kb-section-title"><i class="bi bi-grid-3x3-gap"></i>Concept coverage
          <span class="kb-section-sub">— shared concepts first (top 100)</span>
        </div>
        ${rows.length
          ? `<div class="kb-matrix-wrap"><table class="table table-sm table-hover align-middle">
               <thead><tr>${header}</tr></thead><tbody>${conceptRows}</tbody></table></div>`
          : `<div class="kb-empty-card">No concepts with provenance yet.</div>`}
      </div>`;
}

function exportOverlap() {
    const report = state.overlap;
    if (!report) {
        showToast({ type: "warning", title: "Nothing to export", message: "Refresh the overlap report first." });
        return;
    }
    const rows = (report.overlap || []).map(o => ({
        Subject: o.source,
        Predicate: o.predicate,
        Object: o.target,
        Verdict: o.verdict || "AGREEMENT",
        Modalities: (o.modalities || []).join(", "),
        Documents: (o.docs || []).join(", "),
        Evidence: (o.records || []).map(r => `[${r.doc}] ${r.quality || r.chunk_excerpt || ""}`).join("\n"),
    }));
    const result = exportRowsAsXlsx({
        rows,
        sheetName: "Overlap",
        filename: "kbdebugger_overlap.xlsx",
    });
    showToast(result.ok
        ? { type: "success", title: "Exported", message: `${result.count} overlap rows downloaded.` }
        : { type: "warning", title: "Nothing to export", message: result.reason });
}

/* ----------------------------- Alignment ----------------------------- */

async function scanAlignment() {
    const btn = el("compare-alignment-scan");
    if (btn) btn.disabled = true;
    setStatus("compare-alignment-status", "Embedding node names...");
    try {
        const start = await startAlignmentScan({});
        const result = await pollJob(start.job_id, {
            statusId: "compare-alignment-status",
            label: "Alignment scan",
        });
        state.alignment = Array.isArray(result?.candidates) ? result.candidates : [];
        renderAlignment();
        const hint = state.alignment.length === 0 ? notEnoughGraphDocsHint("align") : "";
        setStatus("compare-alignment-status", hint || `${state.alignment.length} candidate pair(s).`);
    } catch (e) {
        setStatus("compare-alignment-status", `Failed: ${e.message || e}`);
    } finally {
        if (btn) btn.disabled = false;
    }
}

function renderAlignment() {
    const wrap = el("compare-alignment-wrap");
    if (!wrap) return;

    const rows = state.alignment.map((c, idx) => `
      <tr data-idx="${idx}">
        <td class="fw-semibold">${escapeHtml(c.term_a)}</td>
        <td class="fw-semibold">${escapeHtml(c.term_b)}</td>
        <td><span class="badge text-bg-secondary">${Number(c.score).toFixed(2)}</span></td>
        <td class="text-end">
          <div class="btn-group btn-group-sm">
            <button type="button" class="btn btn-outline-success align-accept" title="Same concept">
              <i class="bi bi-check-lg"></i> Same
            </button>
            <button type="button" class="btn btn-outline-danger align-reject" title="Different concepts">
              <i class="bi bi-x-lg"></i> Different
            </button>
          </div>
        </td>
      </tr>
    `);

    wrap.innerHTML = simpleTable(["Term A", "Term B", "Similarity", "Decision"], rows, {
        emptyText: "No pending candidates. Run a scan (after ingesting at least two documents).",
    });

    wrap.querySelectorAll("tr[data-idx]").forEach(tr => {
        const idx = Number(tr.dataset.idx);
        const candidate = state.alignment[idx];
        if (!candidate) return;

        const decide = async (accept) => {
            tr.querySelectorAll("button").forEach(b => { b.disabled = true; });
            try {
                await postAlignmentDecision({
                    term_a: candidate.term_a,
                    term_b: candidate.term_b,
                    accept,
                    score: candidate.score,
                });
                state.alignment = state.alignment.filter(c => c !== candidate);
                renderAlignment();
                setStatus("compare-alignment-status", `${state.alignment.length} candidate pair(s) left.`);
            } catch (e) {
                showToast({ type: "error", title: "Decision failed", message: e.message || String(e) });
                tr.querySelectorAll("button").forEach(b => { b.disabled = false; });
            }
        };

        tr.querySelector(".align-accept")?.addEventListener("click", () => decide(true));
        tr.querySelector(".align-reject")?.addEventListener("click", () => decide(false));
    });
}

/* ----------------------------- Conflicts ----------------------------- */

async function scanConflicts() {
    const btn = el("compare-conflicts-scan");
    if (btn) btn.disabled = true;
    const judge = el("compare-conflicts-judge")?.checked !== false;
    const sources = activeSources();
    setStatus("compare-conflicts-status", "Scanning provenance layer...");
    try {
        const start = await startConflictScan({ judge, sources });
        const result = await pollJob(start.job_id, {
            statusId: "compare-conflicts-status",
            label: "Conflict scan",
        });
        state.conflicts = Array.isArray(result?.conflicts) ? result.conflicts : [];
        renderConflicts();
        const filterNote = sources ? ` (filtered to ${sources.length} source(s))` : "";
        const hint = state.conflicts.length === 0 ? notEnoughGraphDocsHint("compare") : "";
        setStatus("compare-conflicts-status", hint || `${state.conflicts.length} candidate(s)${filterNote}.`);
        await refreshRecordedConflicts();
    } catch (e) {
        setStatus("compare-conflicts-status", `Failed: ${e.message || e}`);
    } finally {
        if (btn) btn.disabled = false;
    }
}

function conflictSideHtml(side) {
    const doc = escapeHtml(side?.doc || "?");
    const text = escapeHtml(side?.text || "");
    const modality = String(side?.modality || "").trim();
    const modalityBadge = modality
        ? ` <span class="badge text-bg-primary">${escapeHtml(modality.toLowerCase())}</span>`
        : "";
    return `<div><span class="badge text-bg-info">${doc}</span>${modalityBadge}<div class="small mt-1">${text}</div></div>`;
}

function renderConflicts() {
    const wrap = el("compare-conflicts-wrap");
    if (!wrap) return;

    const rows = state.conflicts.map((c, idx) => `
      <tr data-idx="${idx}">
        <td><span class="badge text-bg-dark">${escapeHtml(conflictTypeLabel(c.type))}</span></td>
        <td style="max-width: 320px;">${conflictSideHtml(c.side_a)}</td>
        <td style="max-width: 320px;">${conflictSideHtml(c.side_b)}</td>
        <td style="max-width: 260px;">
          <span class="badge ${verdictBadgeClass(c.verdict)}">${escapeHtml(c.verdict || "UNJUDGED")}</span>
          <div class="small text-muted mt-1">${escapeHtml(c.rationale || "")}</div>
        </td>
        <td class="text-end">
          <div class="btn-group btn-group-sm">
            <button type="button" class="btn btn-outline-success conflict-accept" title="Confirm as finding">
              <i class="bi bi-check-lg"></i>
            </button>
            <button type="button" class="btn btn-outline-secondary conflict-dismiss" title="Dismiss">
              <i class="bi bi-x-lg"></i>
            </button>
          </div>
        </td>
      </tr>
    `);

    wrap.innerHTML = simpleTable(["Type", "Document A", "Document B", "Verdict", "Decision"], rows, {
        emptyText: "No conflict candidates. Run a scan after ingesting at least two documents (modality conflicts also need normative statements).",
    });

    wrap.querySelectorAll("tr[data-idx]").forEach(tr => {
        const idx = Number(tr.dataset.idx);
        const candidate = state.conflicts[idx];
        if (!candidate) return;

        const decide = async (accept) => {
            tr.querySelectorAll("button").forEach(b => { b.disabled = true; });
            try {
                await postConflictDecision({ candidate, accept });
                state.conflicts = state.conflicts.filter(c => c !== candidate);
                renderConflicts();
                setStatus("compare-conflicts-status", `${state.conflicts.length} candidate(s) left.`);
                await refreshRecordedConflicts();
            } catch (e) {
                showToast({ type: "error", title: "Decision failed", message: e.message || String(e) });
                tr.querySelectorAll("button").forEach(b => { b.disabled = false; });
            }
        };

        tr.querySelector(".conflict-accept")?.addEventListener("click", () => decide(true));
        tr.querySelector(".conflict-dismiss")?.addEventListener("click", () => decide(false));
    });
}

async function refreshRecordedConflicts() {
    const wrap = el("compare-conflicts-recorded-wrap");
    if (!wrap) return;
    try {
        const { conflicts } = await fetchRecordedConflicts();
        const rows = (conflicts || []).map(c => `
          <tr class="${c.status === "dismissed" ? "opacity-50" : ""}">
            <td><span class="badge text-bg-dark">${escapeHtml(conflictTypeLabel(c.type))}</span></td>
            <td><span class="badge ${verdictBadgeClass(c.verdict)}">${escapeHtml(c.verdict || "")}</span></td>
            <td class="small">${escapeHtml(c.summary || "")}</td>
            <td class="small text-muted">${escapeHtml(c.status || "")}</td>
          </tr>
        `);
        wrap.innerHTML = simpleTable(["Type", "Verdict", "Summary", "Status"], rows, {
            emptyText: "No reviewed conflicts yet.",
        });
    } catch (e) {
        wrap.innerHTML = `<div class="text-muted small">Could not load reviewed conflicts: ${escapeHtml(e.message || String(e))}</div>`;
    }
}

function exportConflicts() {
    const rows = state.conflicts.map(c => ({
        Type: conflictTypeLabel(c.type),
        Summary: c.summary || "",
        "Document A": c.side_a?.doc || "",
        "Statement A": c.side_a?.text || "",
        "Modality A": c.side_a?.modality || "",
        "Document B": c.side_b?.doc || "",
        "Statement B": c.side_b?.text || "",
        "Modality B": c.side_b?.modality || "",
        Verdict: c.verdict || "",
        Rationale: c.rationale || "",
    }));
    const result = exportRowsAsXlsx({
        rows,
        sheetName: "Conflicts",
        filename: "kbdebugger_conflicts.xlsx",
    });
    showToast(result.ok
        ? { type: "success", title: "Exported", message: `${result.count} conflict rows downloaded.` }
        : { type: "warning", title: "Nothing to export", message: result.reason });
}

/* ----------------------------- Ambiguity ----------------------------- */

async function refreshAmbiguity() {
    setStatus("compare-ambiguity-status", "Loading ambiguity report...");
    const sources = activeSources();
    try {
        state.ambiguity = await fetchAmbiguityReport({ sources });
        renderAmbiguity();
        const filterNote = sources ? ` (filtered to ${sources.length} source(s))` : "";
        setStatus("compare-ambiguity-status", filterNote ? filterNote.trim() : "");
    } catch (e) {
        setStatus("compare-ambiguity-status", `Failed: ${e.message || e}`);
    }
}

function renderAmbiguity() {
    const wrap = el("compare-ambiguity-wrap");
    if (!wrap) return;
    const report = state.ambiguity || {};

    const undef = Array.isArray(report.undefined_normative_terms) ? report.undefined_normative_terms : [];
    const undefRows = undef.map(r => `
      <tr>
        <td><span class="badge text-bg-info">${escapeHtml(r.doc)}</span></td>
        <td class="fw-semibold">${escapeHtml(r.term)}</td>
        <td class="text-muted">${escapeHtml(formatPredicateLabel(r.predicate))}</td>
        <td class="small">${escapeHtml(r.example || "")}</td>
      </tr>
    `);

    const vague = Array.isArray(report.vague_language) ? report.vague_language : [];
    const vagueRows = vague.map(r => `
      <tr>
        <td><span class="badge text-bg-info">${escapeHtml(r.doc)}</span></td>
        <td class="fw-semibold">${escapeHtml(r.term)}</td>
        <td>${r.count}</td>
        <td class="small">${(r.examples || []).map(x => escapeHtml(x)).join("<hr class='my-1'>")}</td>
      </tr>
    `);

    const near = Array.isArray(report.near_synonyms) ? report.near_synonyms : [];
    const nearRows = near.map(r => `
      <tr>
        <td class="fw-semibold">${escapeHtml(r.term_a)}</td>
        <td class="fw-semibold">${escapeHtml(r.term_b)}</td>
        <td><span class="badge text-bg-secondary">${Number(r.score).toFixed(2)}</span></td>
      </tr>
    `);

    wrap.innerHTML = `
      <div class="fw-semibold small text-muted mb-1">Obligated but undefined terms (used in requirements, never defined in the same document)</div>
      ${simpleTable(["Document", "Term", "Predicate", "Example statement"], undefRows, {
        emptyText: "No undefined normative terms found (or no normative statements ingested yet).",
    })}
      <div class="fw-semibold small text-muted mb-1 mt-3">Vague / hedging language</div>
      ${simpleTable(["Document", "Term", "Count", "Examples"], vagueRows, {
        emptyText: "No vague-language hits in stored statements.",
    })}
      <div class="fw-semibold small text-muted mb-1 mt-3">Near-synonyms the reviewer kept separate (potential cross-document ambiguity)</div>
      ${simpleTable(["Term A", "Term B", "Similarity"], nearRows, {
        emptyText: "No high-similarity rejected pairs yet (decide some alignment candidates first).",
    })}
    `;
}

function exportAmbiguity() {
    const report = state.ambiguity;
    if (!report) {
        showToast({ type: "warning", title: "Nothing to export", message: "Refresh the ambiguity report first." });
        return;
    }
    const rows = [
        ...(report.undefined_normative_terms || []).map(r => ({
            Kind: "Undefined normative term",
            Document: r.doc,
            Term: r.term,
            Detail: r.predicate,
            Evidence: r.example || "",
        })),
        ...(report.vague_language || []).map(r => ({
            Kind: "Vague language",
            Document: r.doc,
            Term: r.term,
            Detail: `count=${r.count}`,
            Evidence: (r.examples || []).join("\n"),
        })),
        ...(report.near_synonyms || []).map(r => ({
            Kind: "Near-synonym ambiguity",
            Document: "",
            Term: `${r.term_a} / ${r.term_b}`,
            Detail: `similarity=${Number(r.score).toFixed(2)}`,
            Evidence: "",
        })),
    ];
    const result = exportRowsAsXlsx({
        rows,
        sheetName: "Ambiguity",
        filename: "kbdebugger_ambiguity.xlsx",
    });
    showToast(result.ok
        ? { type: "success", title: "Exported", message: `${result.count} ambiguity rows downloaded.` }
        : { type: "warning", title: "Nothing to export", message: result.reason });
}

/* ------------------------------ wiring ------------------------------ */

export function wireComparisonView() {
    // Source selector
    el("compare-sources-refresh")?.addEventListener("click", loadSources);
    el("compare-sources-select-all")?.addEventListener("click", selectAllSources);
    el("compare-sources-clear")?.addEventListener("click", clearAllSources);

    // Analysis actions
    el("compare-overlap-refresh")?.addEventListener("click", refreshOverlap);
    el("compare-overlap-export")?.addEventListener("click", exportOverlap);
    el("compare-alignment-scan")?.addEventListener("click", scanAlignment);
    el("compare-conflicts-scan")?.addEventListener("click", scanConflicts);
    el("compare-conflicts-export")?.addEventListener("click", exportConflicts);
    el("compare-ambiguity-refresh")?.addEventListener("click", refreshAmbiguity);
    el("compare-ambiguity-export")?.addEventListener("click", exportAmbiguity);

    // Load sources immediately on app start (Neo4j + any saved session state)
    loadSources();

    // Re-load when the Compare tab is shown (picks up changes since last visit)
    el("compare-view-tab")?.addEventListener("shown.bs.tab", () => {
        loadSources();
        if (!state.overlapLoaded) {
            refreshOverlap();
            refreshRecordedConflicts();
        }
    });
}

/**
 * Call this after a pipeline run completes so the session section
 * immediately reflects the newly processed documents.
 */
export function refreshComparisonSources() {
    loadSources();
}
