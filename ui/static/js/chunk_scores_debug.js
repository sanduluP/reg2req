/**
 * Chunk scores — DEBUG / TEST FEATURE (safe to remove)
 * ----------------------------------------------------
 * Shows, per dimension, what KeyBERT score each document chunk got against the
 * keyword and whether it passed the paragraph-relevance threshold. Lets you
 * tune thresholds and confirm a (e.g. security) extractor is grabbing the right
 * chunks.
 *
 * To remove this feature later: delete this file, the #chunk-scores-debug
 * container in index.html, its import/call in main.js, and the `chunk_scores`
 * additions in keyword_extraction (scored_chunks) + pipeline_runner.
 */

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
    threshold: null,
    activeDim: null,
    sortDesc: true,
    onlyUnmatched: false,
};

/**
 * Render the debug panel from a pipeline job result.
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
    state.threshold = typeof keybert.para_threshold === "number" ? keybert.para_threshold : null;
    state.activeDim = dims.includes(state.activeDim) ? state.activeDim : dims[0];

    container.classList.remove("d-none");
    render(container);
}

function render(container) {
    const dims = Object.keys(state.byDimension);
    const chunks = (state.byDimension[state.activeDim] || []).slice();

    // Sort: those with a numeric score first (by score), then literal/none.
    chunks.sort((a, b) => {
        const sa = a.score ?? -Infinity;
        const sb = b.score ?? -Infinity;
        return state.sortDesc ? sb - sa : sa - sb;
    });

    const shown = state.onlyUnmatched ? chunks.filter(c => !c.matched) : chunks;
    const matchedCount = chunks.filter(c => c.matched).length;

    const dimOptions = dims.map(d =>
        `<option value="${escapeHtml(d)}" ${d === state.activeDim ? "selected" : ""}>${escapeHtml(d)} (${(state.byDimension[d] || []).length})</option>`
    ).join("");

    const rows = shown.map(c => {
        const kept = c.matched;
        const scoreCell = c.score == null
            ? `<span class="text-muted" title="literal/keyword match — no cosine score">—</span>`
            : `<span class="fw-semibold ${state.threshold != null && c.score >= state.threshold ? "text-success" : "text-muted"}">${c.score.toFixed(3)}</span>`;
        return `
          <tr class="${kept ? "" : "table-light text-muted"}">
            <td>${c.index}</td>
            <td>${scoreCell}</td>
            <td><span class="badge ${kept ? "text-bg-success" : "text-bg-secondary"}">${kept ? "kept" : "dropped"}</span></td>
            <td><span class="badge text-bg-light border">${escapeHtml(c.match_type || "none")}</span></td>
            <td class="small">${escapeHtml(c.excerpt || "")}</td>
          </tr>`;
    }).join("");

    container.innerHTML = `
      <div class="border rounded p-2" style="background:#fffdf5;">
        <div class="d-flex align-items-center gap-2 flex-wrap mb-2">
          <span class="small fw-semibold text-secondary"><i class="bi bi-bug me-1"></i>Chunk scores <span class="badge text-bg-warning">debug</span></span>
          <select id="chunk-scores-dim" class="form-select form-select-sm" style="max-width:260px; font-size:0.74rem;">${dimOptions}</select>
          <span class="small text-muted">${matchedCount}/${chunks.length} kept${state.threshold != null ? ` · threshold ${state.threshold}` : ""}</span>
          <div class="form-check form-check-inline small mb-0 ms-auto">
            <input class="form-check-input" type="checkbox" id="chunk-scores-only-unmatched" ${state.onlyUnmatched ? "checked" : ""}>
            <label class="form-check-label" for="chunk-scores-only-unmatched">dropped only</label>
          </div>
          <button id="chunk-scores-sort" class="btn btn-sm btn-outline-secondary py-0 px-2" style="font-size:0.72rem;">score ${state.sortDesc ? "↓" : "↑"}</button>
        </div>
        <div style="max-height:320px; overflow:auto;">
          <table class="table table-sm table-hover align-middle mb-0" style="font-size:0.78rem;">
            <thead><tr>
              <th style="width:3rem;">#</th><th style="width:5rem;">Score</th>
              <th style="width:5rem;">Kept</th><th style="width:9rem;">Match</th><th>Chunk excerpt</th>
            </tr></thead>
            <tbody>${rows || `<tr><td colspan="5" class="text-muted">No chunks.</td></tr>`}</tbody>
          </table>
        </div>
        <div class="text-muted mt-1" style="font-size:0.66rem;">
          Green score = at/above threshold. This is a temporary inspection view; adjust the threshold in
          “Pipeline thresholds” and re-run to see the cutoff move.
        </div>
      </div>`;

    el("chunk-scores-dim")?.addEventListener("change", (e) => { state.activeDim = e.target.value; render(container); });
    el("chunk-scores-sort")?.addEventListener("click", () => { state.sortDesc = !state.sortDesc; render(container); });
    el("chunk-scores-only-unmatched")?.addEventListener("change", (e) => { state.onlyUnmatched = e.target.checked; render(container); });
}
