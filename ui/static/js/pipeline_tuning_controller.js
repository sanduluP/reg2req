/**
 * Pipeline tuning controller
 * --------------------------
 * Exposes the pipeline's tunable thresholds so the user can "play" with them
 * before a run. Values are read by the pipeline run call via getPipelineThresholds().
 *
 * Controls (each a slider + number box):
 *   - para_threshold : KeyBERT paragraph relevance (how related a paragraph must
 *                      be to the keyword to be decomposed). Higher = stricter.
 *   - sim_threshold  : quality <-> KG subgraph similarity (how close a quality
 *                      must be to existing knowledge to be kept). Higher = stricter.
 *   - top_k          : neighbors retrieved per quality from the KG.
 *   - kg_limit       : KG relations pulled per retrieval pattern.
 *
 * Defaults come from the backend (/api/pipeline/thresholds) so they reflect the
 * env config, and "Reset" restores them.
 */

import { getPipelineThresholds } from "./pipeline_client.js";

const LS_KEY = "kb.pipeline.thresholds";

const SPECS = [
    { key: "para_threshold", label: "Paragraph relevance", min: 0, max: 1, step: 0.01, kind: "float",
      hint: "KeyBERT: how related a paragraph must be to the keyword. Higher = fewer, tighter paragraphs." },
    { key: "sim_threshold", label: "Quality ↔ KG similarity", min: 0, max: 1, step: 0.01, kind: "float",
      hint: "How close a quality must be to existing graph knowledge to be kept. Higher = stricter." },
    { key: "top_k", label: "Neighbors / quality (top-k)", min: 1, max: 25, step: 1, kind: "int",
      hint: "How many nearest KG relations are retrieved per quality." },
    { key: "kg_limit", label: "KG relations / pattern", min: 1, max: 500, step: 1, kind: "int",
      hint: "Cap on relations pulled per retrieval pattern from Neo4j." },
];

const state = {
    defaults: {},   // backend defaults
    values: {},     // current values
    loaded: false,
};

function el(id) { return document.getElementById(id); }

function persist() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(state.values)); } catch (_) { /* ignore */ }
}

function restore() {
    try {
        const raw = localStorage.getItem(LS_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
}

/** Public: the current threshold values sent with a pipeline run. */
export function getPipelineThresholdValues() {
    return { ...state.values };
}

export async function initPipelineTuning({ containerId = "pipeline-tuning" } = {}) {
    const container = el(containerId);
    if (!container) return;

    try {
        state.defaults = await getPipelineThresholds();
    } catch (_) {
        state.defaults = { para_threshold: 0.45, sim_threshold: 0.55, top_k: 5, kg_limit: 50 };
    }

    const saved = restore() || {};
    state.values = {};
    for (const spec of SPECS) {
        const fallback = state.defaults[spec.key];
        const v = saved[spec.key];
        state.values[spec.key] = (v === undefined || v === null || v === "") ? fallback : v;
    }
    state.loaded = true;
    render(container);

    const toggle = el("pipeline-tuning-toggle");
    const panel = el(containerId);
    toggle?.addEventListener("click", () => panel.classList.toggle("d-none"));
}

function render(container) {
    const rows = SPECS.map(spec => {
        const val = state.values[spec.key];
        const isDefault = Number(val) === Number(state.defaults[spec.key]);
        return `
        <div class="pipeline-tuning-row mb-2">
          <div class="d-flex justify-content-between align-items-center" style="font-size:0.74rem;">
            <label class="fw-semibold text-secondary mb-0" title="${spec.hint}">${spec.label}</label>
            <span class="text-muted">${isDefault ? "default" : "custom"} · def ${state.defaults[spec.key]}</span>
          </div>
          <div class="d-flex align-items-center gap-2">
            <input type="range" class="form-range tuning-range" data-key="${spec.key}"
                   min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${val}" style="flex:1;">
            <input type="number" class="form-control form-control-sm tuning-number" data-key="${spec.key}"
                   min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${val}"
                   style="width:5.2rem; font-size:0.74rem;">
          </div>
        </div>`;
    }).join("");

    container.innerHTML = `
      <div class="border rounded p-2" style="background:#f8fafc;">
        <div class="d-flex justify-content-between align-items-center mb-1">
          <span class="small fw-semibold text-secondary"><i class="bi bi-sliders2 me-1"></i>Pipeline thresholds</span>
          <button id="pipeline-tuning-reset" class="btn btn-sm btn-link p-0" type="button" style="font-size:0.72rem;">Reset defaults</button>
        </div>
        ${rows}
        <div class="text-muted" style="font-size:0.66rem;">
          <i class="bi bi-info-circle me-1"></i>Applied to the next run. Stricter thresholds keep less; looser keeps more.
        </div>
      </div>`;

    wire(container);
}

function clampToSpec(spec, raw) {
    let v = Number(raw);
    if (!Number.isFinite(v)) v = Number(state.defaults[spec.key]);
    v = Math.max(spec.min, Math.min(spec.max, v));
    return spec.kind === "int" ? Math.round(v) : v;
}

function wire(container) {
    const sync = (key, raw) => {
        const spec = SPECS.find(s => s.key === key);
        const v = clampToSpec(spec, raw);
        state.values[key] = v;
        // keep both inputs in lockstep
        container.querySelectorAll(`[data-key="${key}"]`).forEach(inp => { inp.value = v; });
        persist();
    };

    container.querySelectorAll(".tuning-range").forEach(r => {
        r.addEventListener("input", () => sync(r.dataset.key, r.value));
    });
    container.querySelectorAll(".tuning-number").forEach(n => {
        n.addEventListener("change", () => { sync(n.dataset.key, n.value); render(container); });
    });

    el("pipeline-tuning-reset")?.addEventListener("click", () => {
        for (const spec of SPECS) state.values[spec.key] = state.defaults[spec.key];
        persist();
        render(container);
    });
}
