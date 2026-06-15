/**
 * UI Reset 🧹
 * ----------
 * Resets the UI when the user decides to nuke the current pipeline session.
 *
 * What it resets:
 * - Upload input (so user can re-upload the same file)
 * - Oversight run context (localStorage + memory)
 * - Oversight stepper UI (back to Step 1)
 * - Oversight controller selections + counters
 * - Extracted triplets cache + UI
 * - Progress section visibility
 */

import { clearRunContext } from "./state/oversight_state.js";
import { resetHumanOversightUI, hasCandidateQualitiesUI } from "./oversight_controller.js";
import { hasTripletsCache, resetExtractedTripletsUI } from "./extracted_triplets_controller.js";
import { setOversightStep, OversightSteps } from "./oversight_stepper.js";

/**
 * Reset the whole "pipeline session".
 *
 * ✅ This is intentionally user-triggered (not automatic after KG upsert),
 * because the reviewer may want to keep inspecting the session state.
 *
 * @param {Object} opts
 * @param {string} [opts.fileInputId="documents"] - File input element id.
 */
export function resetPipelineSession({ fileInputId = "documents" } = {}) {
    // 1) clear provenance/context
    clearRunContext();

    // 2) reset stepper to Step 1 (Candidate Sentences)
    try {
        setOversightStep(OversightSteps.CANDIDATE_QUALITIES);
    } catch (_) {
        // ignore if stepper module isn't present
    }

    // 3) reset controllers UI+state
    resetHumanOversightUI();
    resetExtractedTripletsUI();

    // 4) clear file input (important: allows uploading SAME file again)
    const fileInput = document.getElementById(fileInputId);
    if (fileInput) {
        fileInput.value = "";
        fileInput.blur();
    }

    // 5) hide pipeline progress (since session is nuked)
    const progressSection = document.getElementById("pipeline-progress");
    if (progressSection) progressSection.classList.add("d-none");
}

/**
 * Does the UI currently contain an active "pipeline session" worth warning about?
 *
 * Only in-memory review state counts:
 * - extracted triplets cache exists
 * - candidate sentences are currently rendered in Step 1
 *
 * The persisted run context (localStorage) is intentionally NOT part of this
 * check: it survives page reloads purely as provenance metadata, while the
 * actual session (tables, cached triplets) does not. Warning on a stale
 * context made the confirm dialog appear on every Run click after the first
 * ever run, even when there was nothing on screen to lose.
 *
 * @returns {boolean}
 */
export function hasPipelineSession() {
    return hasTripletsCache() || hasCandidateQualitiesUI();
}
