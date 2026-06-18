/**
 * Extraction settings controller
 * ------------------------------
 * Lets the user choose, before triplet extraction runs:
 *   - a preset (Standards comparison / Pipeline curation / Everything)
 *   - which predicate families are active
 *   - edge-label policy (constrained vs relaxed free-text)
 *   - whether deontic modality is derived/enforced
 *   - extra custom predicates added to any family at run time
 *
 * Settings persist in localStorage and are read by the triplet-extraction call
 * via getExtractionSettings(). The vocabulary itself is fetched from
 * /api/pipeline/triplet-predicates so the UI never hardcodes the predicate list.
 */

import { fetchJson } from "./api_client.js";

const LS_KEY = "kb.extraction.settings";

const state = {
    presets: {},          // {key: {label, families, edge_mode, modality}}
    families: {},         // {familyKey: [predicate, ...]}
    familyLabels: {},     // {familyKey: label}
    defaultPreset: "everything",

    // user selection
    preset: "everything",
    activeFamilies: new Set(),
    edgeMode: "relaxed",
    modalityOn: true,
    customPredicates: [],  // [{predicate, family}]
    loaded: false,
};

function el(id) {
    return document.getElementById(id);
}

function escapeHtml(s) {
    return String(s ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
}

function persist() {
    try {
        localStorage.setItem(LS_KEY, JSON.stringify({
            preset: state.preset,
            activeFamilies: [...state.activeFamilies],
            edgeMode: state.edgeMode,
            modalityOn: state.modalityOn,
            customPredicates: state.customPredicates,
        }));
    } catch (_) { /* ignore */ }
}

function restore() {
    try {
        const raw = localStorage.getItem(LS_KEY);
        if (!raw) return null;
        return JSON.parse(raw);
    } catch (_) {
        return null;
    }
}

/** Apply a preset's defaults to the live selection. */
function applyPreset(presetKey, { keepCustom = true } = {}) {
    const preset = state.presets[presetKey];
    if (!preset) return;
    state.preset = presetKey;
    state.activeFamilies = new Set(preset.families);
    state.edgeMode = preset.edge_mode;
    state.modalityOn = Boolean(preset.modality);
    if (!keepCustom) state.customPredicates = [];
}

/**
 * Public: the settings payload sent to /api/pipeline/triplet-extraction.
 */
export function getExtractionSettings() {
    return {
        preset: state.preset,
        families: [...state.activeFamilies],
        edge_mode: state.edgeMode,
        modality_on: state.modalityOn,
        custom_predicates: state.customPredicates.map(c => c.predicate),
    };
}

export async function initExtractionSettings({ containerId = "extraction-settings" } = {}) {
    const container = el(containerId);
    if (!container) return;

    try {
        const data = await fetchJson("/api/pipeline/triplet-predicates");
        state.presets = data.presets || {};
        state.families = data.families || {};
        state.familyLabels = data.family_labels || {};
        state.defaultPreset = data.default_preset || "everything";
    } catch (e) {
        container.innerHTML = `<div class="text-muted small">Could not load extraction settings: ${escapeHtml(e.message || e)}</div>`;
        return;
    }

    // Restore prior selection, else seed from default preset.
    const saved = restore();
    if (saved && state.presets[saved.preset]) {
        state.preset = saved.preset;
        state.activeFamilies = new Set(
            (saved.activeFamilies || []).filter(f => f in state.families)
        );
        state.edgeMode = saved.edgeMode === "constrained" ? "constrained" : "relaxed";
        state.modalityOn = saved.modalityOn !== false;
        state.customPredicates = Array.isArray(saved.customPredicates)
            ? saved.customPredicates.filter(c => c && c.predicate)
            : [];
        if (!state.activeFamilies.size) applyPreset(state.preset);
    } else {
        applyPreset(state.defaultPreset);
    }

    state.loaded = true;
    render(container);
}

function render(container) {
    const presetButtons = Object.entries(state.presets).map(([key, p]) => `
        <button type="button" data-preset="${escapeHtml(key)}"
            class="btn btn-sm ${key === state.preset ? "btn-primary" : "btn-outline-primary"}"
            style="font-size:0.74rem;">${escapeHtml(p.label)}</button>
    `).join("");

    const familyRows = Object.keys(state.families).map(fkey => {
        const active = state.activeFamilies.has(fkey);
        const label = state.familyLabels[fkey] || fkey;
        const builtins = state.families[fkey] || [];
        const customForFamily = state.customPredicates.filter(c => c.family === fkey);
        const chips = [
            ...builtins.map(p => `<span class="badge text-bg-light border" style="font-weight:400;">${escapeHtml(p)}</span>`),
            ...customForFamily.map(c => `<span class="badge text-bg-warning" title="custom" data-remove-custom="${escapeHtml(c.predicate)}" style="cursor:pointer;font-weight:400;">${escapeHtml(c.predicate)} &times;</span>`),
        ].join(" ");
        return `
          <div class="border rounded p-2 mb-1" style="background:#fff;">
            <div class="form-check mb-1">
              <input class="form-check-input" type="checkbox" id="fam-${escapeHtml(fkey)}" data-family="${escapeHtml(fkey)}" ${active ? "checked" : ""}>
              <label class="form-check-label small fw-semibold" for="fam-${escapeHtml(fkey)}">
                ${escapeHtml(label)} <span class="text-muted">(${builtins.length})</span>
              </label>
            </div>
            <div class="d-flex flex-wrap gap-1" style="font-size:0.68rem;">${chips}</div>
            <div class="input-group input-group-sm mt-2" style="max-width:340px;">
              <input type="text" class="form-control form-control-sm custom-pred-input"
                     data-family="${escapeHtml(fkey)}" placeholder="add custom predicate (e.g. EnforcesPolicy)" style="font-size:0.72rem;">
              <button class="btn btn-outline-secondary add-custom-pred" type="button"
                      data-family="${escapeHtml(fkey)}" style="font-size:0.72rem;">Add</button>
            </div>
          </div>`;
    }).join("");

    container.innerHTML = `
      <div class="d-flex align-items-center gap-2 flex-wrap mb-2">
        <span class="small fw-semibold text-secondary"><i class="bi bi-sliders me-1"></i>Extraction profile</span>
        <div class="btn-group btn-group-sm" role="group">${presetButtons}</div>
        <button id="extraction-settings-advanced-toggle" class="btn btn-sm btn-link p-0 ms-1" type="button" style="font-size:0.74rem;">
          Advanced ▾
        </button>
      </div>

      <div class="d-flex align-items-center gap-3 flex-wrap mb-1">
        <div class="d-flex align-items-center gap-1">
          <span class="small text-muted">Edge labels:</span>
          <div class="btn-group btn-group-sm" role="group">
            <button type="button" data-edge="constrained" class="btn btn-sm ${state.edgeMode === "constrained" ? "btn-secondary" : "btn-outline-secondary"}" style="font-size:0.72rem;">Constrained</button>
            <button type="button" data-edge="relaxed" class="btn btn-sm ${state.edgeMode === "relaxed" ? "btn-secondary" : "btn-outline-secondary"}" style="font-size:0.72rem;">Relaxed (free-text)</button>
          </div>
        </div>
        <div class="form-check form-switch mb-0">
          <input class="form-check-input" type="checkbox" id="extraction-modality-switch" ${state.modalityOn ? "checked" : ""}>
          <label class="form-check-label small" for="extraction-modality-switch">Classify modality (shall/should/may)</label>
        </div>
        <span class="text-muted" style="font-size:0.68rem;">${escapeHtml(summaryLine())}</span>
      </div>

      <div id="extraction-settings-advanced" class="mt-2 d-none">
        <div class="small text-muted mb-1">Active families (a preset preselects these; tick to customize). Add custom predicates to any family — they are merged into the allowed list for the run.</div>
        ${familyRows}
      </div>
    `;

    wireEvents(container);
}

function summaryLine() {
    const famCount = state.activeFamilies.size;
    const custom = state.customPredicates.length;
    return `${famCount} famil${famCount === 1 ? "y" : "ies"} · ${state.edgeMode} · modality ${state.modalityOn ? "on" : "off"}${custom ? ` · ${custom} custom` : ""}`;
}

function wireEvents(container) {
    container.querySelectorAll("[data-preset]").forEach(btn => {
        btn.addEventListener("click", () => {
            applyPreset(btn.dataset.preset);
            persist();
            render(container);
        });
    });

    container.querySelectorAll("[data-edge]").forEach(btn => {
        btn.addEventListener("click", () => {
            state.edgeMode = btn.dataset.edge;
            persist();
            render(container);
        });
    });

    const modSwitch = el("extraction-modality-switch");
    modSwitch?.addEventListener("change", () => {
        state.modalityOn = modSwitch.checked;
        persist();
        render(container);
    });

    const advToggle = el("extraction-settings-advanced-toggle");
    const advPanel = el("extraction-settings-advanced");
    advToggle?.addEventListener("click", () => {
        advPanel?.classList.toggle("d-none");
    });

    container.querySelectorAll("[data-family]").forEach(input => {
        if (input.type !== "checkbox") return;
        input.addEventListener("change", () => {
            const f = input.dataset.family;
            if (input.checked) state.activeFamilies.add(f);
            else state.activeFamilies.delete(f);
            persist();
            // re-render to refresh summary, keep advanced open
            const wasOpen = !advPanel?.classList.contains("d-none");
            render(container);
            if (wasOpen) el("extraction-settings-advanced")?.classList.remove("d-none");
        });
    });

    container.querySelectorAll(".add-custom-pred").forEach(btn => {
        btn.addEventListener("click", () => {
            const family = btn.dataset.family;
            const input = container.querySelector(`.custom-pred-input[data-family="${family}"]`);
            const value = (input?.value || "").trim();
            if (!value) return;
            // Avoid duplicates (case-insensitive against builtins + customs).
            const existing = new Set([
                ...Object.values(state.families).flat().map(p => p.toLowerCase()),
                ...state.customPredicates.map(c => c.predicate.toLowerCase()),
            ]);
            if (!existing.has(value.toLowerCase())) {
                state.customPredicates.push({ predicate: value, family });
                // Make sure the family is active so the custom predicate is used.
                state.activeFamilies.add(family);
            }
            persist();
            render(container);
            el("extraction-settings-advanced")?.classList.remove("d-none");
        });
    });

    container.querySelectorAll("[data-remove-custom]").forEach(chip => {
        chip.addEventListener("click", () => {
            const pred = chip.dataset.removeCustom;
            state.customPredicates = state.customPredicates.filter(c => c.predicate !== pred);
            persist();
            render(container);
            el("extraction-settings-advanced")?.classList.remove("d-none");
        });
    });
}
