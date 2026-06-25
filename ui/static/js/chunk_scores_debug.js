/**
 * Chunk scores — interactive relevance tuning
 * -------------------------------------------
 * After a run, shows EVERY chunk's similarity score against the keyword. A
 * threshold slider re-classifies chunks as kept/dropped LIVE (no re-run), and
 * the chosen value is pushed to "Pipeline thresholds" so the next run uses it.
 *
 * Originally a debug view; safe to remove by deleting this file, the
 * #chunk-scores-debug container in index.html, its import/call in main.js, and
 * the scored_chunks/chunk_scores additions in keyword_extraction + pipeline_runner.
 */

import { setPipelineThreshold } from "./pipeline_tuning_controller.js";

function el(id) { return document.getElementById(id); }

function escapeHtml(s) {
    return String(s ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
}

const state = {
    byDimension: {},     // {dim: [chunk, ...]}
    threshold: 0.45,
    activeDim: null,
    sortDesc: true,
};

/** Literal/keyword matches are always kept; semantic matches depend on the threshold. */
function isLiteral(chunk) {
    return chunk.match_type === "exact" || chunk.match_type === "synonym";
}

function isKept(chunk, threshold) {
    if (isLiteral(chunk)) return true;
    return typeof chunk.score === "number" && chunk.score >= threshold;
}

/**
 * Render the panel from a pipeline job result.
 * @param {object} pipelineResult - the /run job result (has KeyBERT.chunk_scores)
 */
/** Clear + hide the chunk-relevance panel (used when a new run starts). */
export function clearChunkScoresDebug() {
    const container = el("chunk-scores-debug");
    const wrap = el("chunk-scores-wrap");
    state.byDimension = {};
    state.activeDim = null;
    if (container) container.innerHTML = "";
    if (wrap) wrap.classList.add("d-none");
}

export function renderChunkScoresDebug(pipelineResult) {
    const container = el("chunk-scores-debug");
    const wrap = el("chunk-scores-wrap");
    if (!container) return;

    const keybert = pipelineResult?.KeyBERT || {};
    const byDim = keybert.chunk_scores || {};
    const dims = Object.keys(byDim);

    if (!dims.length) {
        if (wrap) wrap.classList.add("d-none");
        container.innerHTML = "";
        return;
    }

    state.byDimension = byDim;
    state.threshold = typeof keybert.para_threshold === "number" ? keybert.para_threshold : 0.45;
    state.activeDim = dims.includes(state.activeDim) ? state.activeDim : dims[0];

    if (wrap) wrap.classList.remove("d-none");

    // Auto-expand the collapse when new scores arrive
    const collapse = el("chunk-scores-collapse");
    if (collapse && !collapse.classList.contains("show")) {
        window.bootstrap?.Collapse?.getOrCreateInstance(collapse)?.show();
    }

    render(container);
}

function render(container) {
    const dims = Object.keys(state.byDimension);
    const chunks = (state.byDimension[state.activeDim] || []).slice();

    chunks.sort((a, b) => {
        const sa = a.score ?? -Infinity;
        const sb = b.score ?? -Infinity;
        return state.sortDesc ? sb - sa : sa - sb;
    });

    const keptCount = chunks.filter(c => isKept(c, state.threshold)).length;

    // Keep the collapsible toggle header in sync with the live count
    const toggleInfo = el("chunk-scores-toggle-info");
    if (toggleInfo) toggleInfo.textContent = `${keptCount}/${chunks.length} included`;

    const dimOptions = dims.map(d =>
        `<option value="${escapeHtml(d)}" ${d === state.activeDim ? "selected" : ""}>${escapeHtml(d)} (${(state.byDimension[d] || []).length})</option>`
    ).join("");

    const rows = chunks.map(c => {
        const kept = isKept(c, state.threshold);
        const scoreNum = typeof c.score === "number" ? c.score : null;
        const scoreCell = scoreNum == null
            ? `<span class="text-muted">n/a</span>`
            : `<span class="fw-semibold ${kept ? "text-success" : "text-muted"}">${scoreNum.toFixed(3)}</span>`;
        const statusBadge = kept
            ? `<span class="badge text-bg-success">included</span>`
            : `<span class="badge text-bg-secondary">excluded</span>`;
        const literalTag = isLiteral(c)
            ? ` <span class="badge text-bg-light border" title="literal/keyword match — always included">${escapeHtml(c.match_type)}</span>`
            : "";
        return `
          <tr class="${kept ? "" : "table-light text-muted"}">
            <td>${c.index}</td>
            <td>${scoreCell}</td>
            <td>${statusBadge}${literalTag}</td>
            <td class="small">${escapeHtml(c.excerpt || "")}</td>
          </tr>`;
    }).join("");

    const sortArrow = state.sortDesc ? "↓" : "↑";

    container.innerHTML = `
      <div class="chunk-rel-panel">
        <div class="chunk-rel-controls">
          <select id="chunk-scores-dim" class="form-select form-select-sm chunk-rel-dim" title="Focus area">${dimOptions}</select>
          <div class="chunk-rel-thresh">
            <span class="chunk-rel-thresh-label">Threshold</span>
            <input type="range" class="form-range" id="chunk-scores-threshold"
                   min="0" max="1" step="0.01" value="${state.threshold}">
            <input type="number" class="form-control form-control-sm" id="chunk-scores-threshold-num"
                   min="0" max="1" step="0.01" value="${state.threshold}">
          </div>
        </div>

        <div class="chunk-rel-table">
          <table class="table table-sm table-hover align-middle mb-0">
            <thead><tr>
              <th style="width:2.4rem;">#</th>
              <th style="width:5rem; cursor:pointer; user-select:none;" id="chunk-scores-sort" title="Sort by score">Score ${sortArrow}</th>
              <th style="width:6rem;">Status</th>
              <th>Excerpt</th>
            </tr></thead>
            <tbody>${rows || `<tr><td colspan="4" class="text-muted">No chunks.</td></tr>`}</tbody>
          </table>
        </div>
        <div class="chunk-rel-hint"><i class="bi bi-info-circle me-1"></i>Drag the threshold to include/exclude chunks — applied to your next run.</div>
      </div>`;

    wire(container);
}

function applyThreshold(container, value) {
    let v = Number(value);
    if (!Number.isFinite(v)) return;
    v = Math.max(0, Math.min(1, v));
    state.threshold = v;
    // Keep both inputs in sync and push to the pipeline tuning panel.
    const r = el("chunk-scores-threshold");
    const n = el("chunk-scores-threshold-num");
    if (r) r.value = v;
    if (n) n.value = v;
    setPipelineThreshold("para_threshold", v);
    // Re-render rows + summary (cheap, client-side).
    render(container);
    // Notify the oversight view so it can live-filter the extracted qualities
    // by this same relevance threshold (debounced on the listener side).
    document.dispatchEvent(new CustomEvent("kb:chunk-threshold-change", { detail: { threshold: v } }));
}

function wire(container) {
    el("chunk-scores-dim")?.addEventListener("change", (e) => { state.activeDim = e.target.value; render(container); });
    el("chunk-scores-sort")?.addEventListener("click", () => { state.sortDesc = !state.sortDesc; render(container); });
    el("chunk-scores-threshold")?.addEventListener("input", (e) => applyThreshold(container, e.target.value));
    el("chunk-scores-threshold-num")?.addEventListener("change", (e) => applyThreshold(container, e.target.value));
}
