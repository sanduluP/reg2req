/**
 * Chunk scores — interactive relevance tuning
 * -------------------------------------------
 * After a run, shows EVERY chunk's similarity score against the keyword. A
 * threshold slider re-classifies chunks as selected/dropped LIVE (no re-run),
 * and the chosen value is pushed to "Pipeline thresholds" so the next run uses
 * it. It also reacts to changes made directly in the Pipeline thresholds panel.
 *
 * Safe to remove: delete this file, the #chunk-scores-debug container in
 * index.html, its import/call in main.js, and the scored_chunks/chunk_scores
 * additions in keyword_extraction + pipeline_runner.
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

/** Literal/keyword matches are always selected, regardless of cosine score. */
function isLiteral(chunk) {
    return chunk.match_type === "exact" || chunk.match_type === "synonym";
}

function isSelected(chunk, threshold) {
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

// Keep in sync when the threshold is changed from the Pipeline thresholds panel.
window.addEventListener("kb:pipeline-threshold-change", (event) => {
    const threshold = event.detail?.values?.para_threshold;
    if (typeof threshold !== "number" || Number.isNaN(threshold)) return;
    state.threshold = threshold;
    const container = el("chunk-scores-debug");
    if (container && !container.classList.contains("d-none")) render(container);
});

function render(container) {
    const dims = Object.keys(state.byDimension);
    const chunks = (state.byDimension[state.activeDim] || []).slice();

    chunks.sort((a, b) => {
        const sa = a.score ?? -Infinity;
        const sb = b.score ?? -Infinity;
        return state.sortDesc ? sb - sa : sa - sb;
    });

    const selectedCount = chunks.filter(c => isSelected(c, state.threshold)).length;

    const dimOptions = dims.map(d =>
        `<option value="${escapeHtml(d)}" ${d === state.activeDim ? "selected" : ""}>${escapeHtml(d)} (${(state.byDimension[d] || []).length})</option>`
    ).join("");

    const rows = chunks.map(c => {
        const selected = isSelected(c, state.threshold);
        const scoreNum = typeof c.score === "number" ? c.score : null;
        const scoreCell = scoreNum == null
            ? `<span class="text-muted">n/a</span>`
            : `<span class="fw-semibold ${selected ? "text-success" : "text-muted"}">${scoreNum.toFixed(3)}</span>`;
        const statusBadge = selected
            ? `<span class="badge text-bg-success">Selected</span>`
            : `<span class="badge text-bg-secondary">Dropped</span>`;
        return `
          <tr class="${selected ? "" : "table-light text-muted"}">
            <td>${c.index}</td>
            <td>${scoreCell}</td>
            <td>${statusBadge}</td>
            <td><span class="badge text-bg-light border">${escapeHtml(c.match_type || "none")}</span></td>
            <td class="small">${escapeHtml(c.excerpt || "")}</td>
          </tr>`;
    }).join("");

    container.innerHTML = `
      <div class="border rounded p-2" style="background:#fffdf5;">
        <div class="d-flex align-items-center gap-2 flex-wrap mb-2">
          <span class="small fw-semibold text-secondary"><i class="bi bi-funnel me-1"></i>Chunk relevance</span>
          <select id="chunk-scores-dim" class="form-select form-select-sm" style="max-width:260px; font-size:0.74rem;">${dimOptions}</select>
          <span class="small text-muted" id="chunk-scores-summary">${selectedCount}/${chunks.length} selected · threshold ${state.threshold.toFixed(2)}</span>
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
              <th style="width:8rem;">
                <div class="d-flex align-items-center gap-2">
                  <span>Score</span>
                  <button id="chunk-scores-sort" class="btn btn-sm btn-outline-secondary py-0 px-2" type="button" style="font-size:0.72rem;">
                    ${state.sortDesc ? "↓" : "↑"}
                  </button>
                </div>
              </th>
              <th style="width:6rem;">Status</th>
              <th style="width:9rem;">Match</th>
              <th>Chunk excerpt</th>
            </tr></thead>
            <tbody>${rows || `<tr><td colspan="5" class="text-muted">No chunks.</td></tr>`}</tbody>
          </table>
        </div>
        <div class="text-muted mt-1" style="font-size:0.66rem;">
          <i class="bi bi-info-circle me-1"></i>Drag the threshold to select/drop chunks live (literal keyword matches stay selected).
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
    const r = el("chunk-scores-threshold");
    const n = el("chunk-scores-threshold-num");
    if (r) r.value = v;
    if (n) n.value = v;
    setPipelineThreshold("para_threshold", v);  // apply to the next run
    render(container);
}

function wire(container) {
    el("chunk-scores-dim")?.addEventListener("change", (e) => { state.activeDim = e.target.value; render(container); });
    el("chunk-scores-sort")?.addEventListener("click", () => { state.sortDesc = !state.sortDesc; render(container); });
    el("chunk-scores-threshold")?.addEventListener("input", (e) => applyThreshold(container, e.target.value));
    el("chunk-scores-threshold-num")?.addEventListener("change", (e) => applyThreshold(container, e.target.value));
}
