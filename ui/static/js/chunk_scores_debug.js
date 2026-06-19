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
export function renderChunkScoresDebug(pipelineResult) {
    const container = el("chunk-scores-debug");
    if (!container) return;

    const keybert = pipelineResult?.KeyBERT || {};
    const byDim = keybert.chunk_scores || {};
    const dims = Object.keys(byDim);

    if (!dims.length) {
        container.classList.add("d-none");
        container.innerHTML = "";
        return;
    }

    state.byDimension = byDim;
    state.threshold = typeof keybert.para_threshold === "number" ? keybert.para_threshold : 0.45;
    state.activeDim = dims.includes(state.activeDim) ? state.activeDim : dims[0];

    container.classList.remove("d-none");
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
      <div class="border rounded p-2" style="background:#fffdf5;">
        <div class="d-flex align-items-center gap-2 flex-wrap mb-2">
          <span class="small fw-semibold text-secondary"><i class="bi bi-funnel me-1"></i>Chunk relevance</span>
          <select id="chunk-scores-dim" class="form-select form-select-sm" style="max-width:260px; font-size:0.74rem;">${dimOptions}</select>
          <span class="small text-muted" id="chunk-scores-summary">${keptCount}/${chunks.length} included</span>
        </div>

        <div class="d-flex align-items-center gap-2 mb-2">
          <label class="small text-secondary mb-0" style="white-space:nowrap;">Relevance threshold</label>
          <input type="range" class="form-range" id="chunk-scores-threshold"
                 min="0" max="1" step="0.01" value="${state.threshold}" style="flex:1; max-width:340px;">
          <input type="number" class="form-control form-control-sm" id="chunk-scores-threshold-num"
                 min="0" max="1" step="0.01" value="${state.threshold}" style="width:5.2rem; font-size:0.74rem;">
        </div>

        <div style="max-height:320px; overflow:auto;">
          <table class="table table-sm table-hover align-middle mb-0" style="font-size:0.78rem;">
            <thead><tr>
              <th style="width:3rem;">#</th>
              <th style="width:7rem; cursor:pointer; user-select:none;" id="chunk-scores-sort" title="Sort by score">Score ${sortArrow}</th>
              <th style="width:11rem;">Status</th>
              <th>Chunk excerpt</th>
            </tr></thead>
            <tbody>${rows || `<tr><td colspan="4" class="text-muted">No chunks.</td></tr>`}</tbody>
          </table>
        </div>
        <div class="text-muted mt-1" style="font-size:0.66rem;">
          <i class="bi bi-info-circle me-1"></i>Drag the threshold to include/exclude chunks live (literal keyword matches stay included).
          This value is applied to your next run.
        </div>
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
}

function wire(container) {
    el("chunk-scores-dim")?.addEventListener("change", (e) => { state.activeDim = e.target.value; render(container); });
    el("chunk-scores-sort")?.addEventListener("click", () => { state.sortDesc = !state.sortDesc; render(container); });
    el("chunk-scores-threshold")?.addEventListener("input", (e) => applyThreshold(container, e.target.value));
    el("chunk-scores-threshold-num")?.addEventListener("change", (e) => applyThreshold(container, e.target.value));
}
