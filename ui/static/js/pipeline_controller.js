/**
 * Orchestrates upload -> start job -> poll -> update UI.
 */

import { startPipelineJob, getJobStatus } from "./pipeline_client.js";
import { hideProgressPanel, showProgressPanel, updateProgressPanel } from "./pipeline_progress_ui.js";
import { resetElapsedTimer, updateElapsedTimer } from "./timer.js";
import { setRunContext } from "./state/oversight_state.js";
import { confirmModal } from "./modals/confirm_modal.js";
import { hasPipelineSession, resetPipelineSession } from "./ui_reset.js";


/**
 * Pipeline Controller 🚀
 * ---------------------
 * Orchestrates:
 * - Validate keyword + file
 * - Optionally confirm overwriting an existing session
 * - Start job
 * - Poll status
 * - Update progress + elapsed timer
 *
 * It does NOT automatically reset session on KG-upsert anymore.
 * The user controls session lifetime via the "Reset current pipeline" action.
 */

/**
 * @typedef {Object} WirePipelineRunControlsParams
 * @property {string} params.fileInputId
 * @property {string} params.keywordSelectId
 * @property {string} params.runBtnId
 * @property {string} params.runMenuBtnId
 * @property {string} params.resetBtnId
 * @property {(result:any)=>void} [params.onDone]
 */

/**
 * Wire the pipeline controls:
 * - Enables/disables Run button based on keyword+file state
 * - Runs pipeline on click (with overwrite confirmation if session exists)
 * - Allows manual reset via split-menu action
 *
 * @param {WirePipelineRunControlsParams} params
 */
export function wirePipelineRunControls({
  fileInputId,
  keywordSelectId,
  runBtnId,
  runMenuBtnId,
  resetBtnId,
  onDone,
}) {
  const fileInput = document.getElementById(fileInputId);
  const keywordSel = document.getElementById(keywordSelectId);

  const runBtn = document.getElementById(runBtnId);
  const runMenuBtn = document.getElementById(runMenuBtnId);
  const resetBtn = document.getElementById(resetBtnId);

  if (!fileInput) throw new Error(`Missing file input: #${fileInputId}`);
  if (!keywordSel) throw new Error(`Missing keyword select: #${keywordSelectId}`);
  if (!runBtn) throw new Error(`Missing run button: #${runBtnId}`);
  if (!runMenuBtn) throw new Error(`Missing run menu button: #${runMenuBtnId}`);
  if (!resetBtn) throw new Error(`Missing reset button: #${resetBtnId}`);

  /** Whether a pipeline run is currently active. */
  let isRunning = false;

  /**
   * Update Run button state + appearance based on:
   * - isRunning
   * - keyword selected
   * - file selected
   */
  function syncRunUi() {
    const keyword = (keywordSel.value || "").trim();
    const hasKeyword = Boolean(keyword);
    const hasFile = Boolean(fileInput.files && fileInput.files.length > 0);

    const enabled = hasKeyword && hasFile && !isRunning;

    runBtn.disabled = !enabled;
    runMenuBtn.disabled = !enabled; // only allow reset when not running (you can change this)
    resetBtn.disabled = isRunning;  // don’t allow reset mid-run

    // Tooltip hints
    if (!hasKeyword) runBtn.title = "🔐 Select a keyword first";
    else if (!hasFile) runBtn.title = "📄 Choose a file first";
    else runBtn.title = "🚀 Start pipeline";

    _setRunButtonVisualState(runBtn, { running: isRunning });
  }

  /**
   * Mark the Run button as running/idle (spinner vs play icon).
   *
   * @param {HTMLButtonElement} btn
   * @param {{running:boolean}} state
   */
  function _setRunButtonVisualState(btn, { running }) {
    const iconWrap = btn.querySelector(".pipeline-run-icon");
    const label = btn.querySelector(".pipeline-run-label");

    if (iconWrap) {
      iconWrap.innerHTML = running
        ? `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>`
        : `<i class="bi bi-play-fill"></i>`;
    }
    if (label) label.textContent = running ? "Running…" : "Run";
  }

  // Recompute enablement when user changes keyword or file.
  keywordSel.addEventListener("change", syncRunUi);
  fileInput.addEventListener("change", syncRunUi);

  // Manual reset action (split-menu)
  resetBtn.addEventListener("click", async () => {
    if (isRunning) return;

    if (!hasPipelineSession()) {
      // nothing to reset, but still give user feedback if you want
      return;
    }

    const ok = await confirmModal({
      title: "🧨 Reset current pipeline session?",
      body:
        "This will clear the current run (oversight tables, cached triplets, and the selected upload file). " +
        "You can't undo this.",
      confirmText: "Yes, reset",
      cancelText: "Cancel",
      confirmBtnClass: "btn-danger",
    });

    if (!ok) return;

    resetPipelineSession();
    syncRunUi();
  });

  // Run pipeline (primary action)
  runBtn.addEventListener("click", async () => {
    if (isRunning) return;

    const keyword = (keywordSel.value || "").trim();
    if (!keyword) return;

    const file = fileInput.files?.[0];
    if (!file) return;

    // If there is an existing session, confirm overwrite
    if (hasPipelineSession()) {
      const ok = await confirmModal({
        title: "⚠️ Start a new pipeline run?",
        body:
          "A previous pipeline session already exists. Starting a new run will erase the current session " +
          "(candidate sentences, extracted triplets, and cached review state).",
        confirmText: "Yes, start new run",
        cancelText: "Cancel",
        confirmBtnClass: "btn-warning",
      });

      if (!ok) return;

      // Nuke old session now (user explicitly agreed)
      resetPipelineSession();
    }

    // --- Start pipeline ---
    isRunning = true;
    syncRunUi();

    showProgressPanel();
    resetElapsedTimer();
    updateProgressPanel({ stage: "queued", message: "Queued…", current: null, total: null });

    let jobId;
    try {
      const startResp = await startPipelineJob({ keyword, file });
      jobId = startResp.job_id;
    } catch (err) {
      // eslint-disable-next-line no-alert
      alert(`Failed to start pipeline: ${err.message}`);
      isRunning = false;
      syncRunUi();
      return;
    }

    const poll = async () => {
      let job;

      try {
        job = await getJobStatus(jobId);
      } catch (err) {
        console.error("Polling request failed:", err);
        alert(`Polling request failed: ${err.message}`);
        isRunning = false;
        syncRunUi();
        return;
      }

      try {
        updateProgressPanel({
          stage: job.stage,
          message: job.message,
          current: job.progress?.current ?? null,
          total: job.progress?.total ?? null,
        });

        updateElapsedTimer(job);

        if (job.state === "done") {
          const meta = job.result?._meta;
          if (meta?.source) {
            setRunContext({
              source: meta.source,
              source_name: meta.source_name || null,
              keyword: meta.keyword || keyword || null,
            });
          }

          isRunning = false;
          syncRunUi();

          hideProgressPanel();
          onDone?.(job.result);
          return;
        }

        if (job.state === "error") {
          alert(`Pipeline failed: ${job.error || "unknown error"}`);
          isRunning = false;
          syncRunUi();
          return;
        }

        setTimeout(poll, 1000);
      } catch (err) {
        console.error("Polling UI/update failed:", err, job);
        alert(`Polling succeeded, but UI update failed: ${err.message}`);
        isRunning = false;
        syncRunUi();
      }
    };


    poll();
  });


  /**
   * Update the small "Ready ✅" badge next to the Run button.
   *
   * Badge is shown if:
   * - keyword is selected
   * - a file is selected
   * - not currently running
   *
   * @param {Object} opts
   * @param {boolean} opts.ready
   */
  function setRunReadyBadge({ ready }) {
    const badge = document.getElementById("run-ready-badge");
    if (!badge) return;
    badge.classList.toggle("d-none", !ready);
  }

  // whenever keyword changes or file changes, call:
  function recomputeReady() {
    const keyword = (keywordSel?.value ?? "").trim();
    const hasFile = Boolean(fileInput?.files && fileInput.files.length > 0);
    const ready = Boolean(keyword && hasFile && !isRunning);
    setRunReadyBadge({ ready });
  }


  // Initial UI state
  syncRunUi();
}
