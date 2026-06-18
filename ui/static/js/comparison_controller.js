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

/* ======================== Source selector ======================== */

async function loadSources() {
    const loadingSpinner = el("compare-sources-loading");
    if (loadingSpinner) loadingSpinner.classList.remove("d-none");

    // 1) Neo4j sources — always try, even before any pipeline run
    let neo4jSources = [];
    try {
        const data = await fetchComparisonSources();
        neo4jSources = Array.isArray(data.neo4j_sources) ? data.neo4j_sources : [];
    } catch (e) {
        console.warn("Could not fetch Neo4j sources:", e);
    }

    // 2) Session sources — documents from the current pipeline run, before Neo4j submission
    const ctx = getRunContext();
    const sessionNames = Array.isArray(ctx?.source_names)
        ? ctx.source_names
        : (ctx?.source_name ? [ctx.source_name] : []);

    const neo4jSet = new Set(neo4jSources);
    // Session docs not yet in the graph get their own section
    const sessionOnly = sessionNames.filter(n => !neo4jSet.has(n));

    // Combined for filter logic: graph first, then session-only
    state.allSources = [
        ...neo4jSources.map(n => ({ name: n, type: "graph" })),
        ...sessionOnly.map(n => ({ name: n, type: "session" })),
    ];
    state.selectedSources = null; // default: all

    _renderGraphSourceSection(neo4jSources);
    _renderSessionSourceSection(sessionOnly, neo4jSources.length > 0);

    if (loadingSpinner) loadingSpinner.classList.add("d-none");
}

function _renderGraphSourceSection(neo4jSources) {
    const wrap = el("compare-graph-sources-wrap");
    const hint = el("compare-graph-sources-hint");
    const count = el("compare-graph-count");

    if (!wrap) return;

    if (count) count.textContent = neo4jSources.length ? String(neo4jSources.length) : "";

    if (neo4jSources.length === 0) {
        wrap.innerHTML = `<span class="text-muted small">No knowledge in graph yet.</span>`;
        if (hint) hint.innerHTML =
            `Run the pipeline on PDFs, then <em>Submit to KG</em> — ` +
            `or use <em>Initialize graph</em> (toolbar step 1) to load baseline knowledge.`;
        return;
    }

    wrap.innerHTML = "";
    if (hint) hint.textContent = "Click a source to toggle it in/out of the analysis.";

    for (const name of neo4jSources) {
        wrap.appendChild(_makeChip(name, "graph"));
    }
}

function _renderSessionSourceSection(sessionOnly, graphHasData) {
    const wrap = el("compare-session-sources-wrap");
    const hint = el("compare-session-sources-hint");
    const count = el("compare-session-count");

    if (!wrap) return;

    if (count) count.textContent = sessionOnly.length ? String(sessionOnly.length) : "";

    if (sessionOnly.length === 0) {
        const ctx = getRunContext();
        const hasSession = ctx?.source_names?.length > 0 || ctx?.source_name;
        if (hasSession) {
            // Session docs exist but ALL are already in the graph
            wrap.innerHTML = `<span class="text-muted small">All session docs already in graph.</span>`;
            if (hint) hint.textContent = "";
        } else {
            wrap.innerHTML = `<span class="text-muted small">No active session.</span>`;
            if (hint) hint.textContent = "Run the pipeline to see current documents here.";
        }
        return;
    }

    wrap.innerHTML = "";
    for (const name of sessionOnly) {
        wrap.appendChild(_makeChip(name, "session"));
    }
    if (hint) hint.innerHTML =
        `Not yet in Neo4j — go to <em>Review &amp; extract</em> → Submit to KG to persist.`;
}

function _makeChip(name, type) {
    const isSelected = state.selectedSources === null || state.selectedSources.has(name);
    const isSession = type === "session";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.sourceName = name;
    btn.dataset.sourceType = type;
    btn.className = [
        "btn btn-sm py-0 px-2",
        isSelected
            ? (isSession ? "btn-warning" : "btn-primary")
            : "btn-outline-secondary",
    ].join(" ");
    btn.style.cssText = "font-size:0.72rem; border-radius:999px; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;";
    btn.title = `${isSession ? "Session" : "Graph"} source: ${name}`;
    btn.textContent = name;

    btn.addEventListener("click", () => {
        if (state.selectedSources === null) {
            state.selectedSources = new Set(state.allSources.map(s => s.name));
        }
        if (state.selectedSources.has(name)) {
            state.selectedSources.delete(name);
        } else {
            state.selectedSources.add(name);
        }
        if (state.selectedSources.size === state.allSources.length) {
            state.selectedSources = null;
        }
        _updateChipStyles();
    });

    return btn;
}

function _updateChipStyles() {
    // Update visual state without re-rendering the whole list
    const allChips = document.querySelectorAll(
        "#compare-graph-sources-wrap button[data-source-name], " +
        "#compare-session-sources-wrap button[data-source-name]"
    );
    allChips.forEach(btn => {
        const name = btn.dataset.sourceName;
        const isSession = btn.dataset.sourceType === "session";
        const isSelected = state.selectedSources === null || state.selectedSources.has(name);
        btn.className = [
            "btn btn-sm py-0 px-2",
            isSelected
                ? (isSession ? "btn-warning" : "btn-primary")
                : "btn-outline-secondary",
        ].join(" ");
    });
}

function selectAllSources() {
    state.selectedSources = null;
    _updateChipStyles();
}

function clearAllSources() {
    state.selectedSources = new Set();
    _updateChipStyles();
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

    // 1) Coverage per document
    const coverage = Array.isArray(report.coverage) ? report.coverage : [];
    const covRows = coverage.map(c => `
      <tr>
        <td class="fw-semibold">${escapeHtml(c.doc)}</td>
        <td>${c.assertions}</td>
        <td>${c.concepts}</td>
        <td>${c.normative_statements}</td>
      </tr>
    `);
    el("compare-coverage-wrap").innerHTML = `
      <div class="fw-semibold small text-muted mb-1">Document coverage</div>
      ${simpleTable(["Document", "Assertions", "Concepts", "Normative statements"], covRows, {
        emptyText: "No provenance-carrying edges in the graph yet. Run the pipeline and submit triplets first.",
    })}
    `;

    // 2) Multi-document assertions (the overlap itself)
    const overlap = Array.isArray(report.overlap) ? report.overlap : [];
    const ovRows = overlap.map(o => `
      <tr>
        <td>${escapeHtml(o.source)}</td>
        <td class="text-muted">${escapeHtml(formatPredicateLabel(o.predicate))}</td>
        <td>${escapeHtml(o.target)}</td>
        <td>${(o.docs || []).map(d => `<span class="badge text-bg-info me-1">${escapeHtml(d)}</span>`).join("")}</td>
      </tr>
    `);
    el("compare-overlap-wrap").innerHTML = `
      <div class="fw-semibold small text-muted mb-1">Assertions supported by multiple documents</div>
      ${simpleTable(["Subject", "Predicate", "Object", "Documents"], ovRows, {
        emptyText: "No assertion is supported by more than one document yet.",
    })}
    `;

    // 3) Concept × document matrix
    const concepts = report.concepts || {};
    const docs = Array.isArray(concepts.documents) ? concepts.documents : [];
    const rows = Array.isArray(concepts.rows) ? concepts.rows : [];
    const conceptRows = rows.slice(0, 100).map(r => `
      <tr>
        <td class="fw-semibold">${escapeHtml(r.concept)}</td>
        ${docs.map(d => {
            const n = r.counts?.[d] || 0;
            return `<td class="${n ? "" : "text-muted"}">${n || "—"}</td>`;
        }).join("")}
        <td>${r.docs}</td>
      </tr>
    `);
    el("compare-concepts-wrap").innerHTML = `
      <div class="fw-semibold small text-muted mb-1">Concept coverage matrix (top 100; shared concepts first)</div>
      ${simpleTable(["Concept", ...docs, "# Docs"], conceptRows, {
        emptyText: "No concepts with provenance yet.",
    })}
    `;
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
        setStatus("compare-alignment-status", `${state.alignment.length} candidate pair(s).`);
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
        setStatus("compare-conflicts-status", `${state.conflicts.length} candidate(s)${filterNote}.`);
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
