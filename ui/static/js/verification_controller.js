/**
 * Verification panel controller (Phase B: 5-strategy graph verification).
 *
 * Runs AFTER the graph is built (auto-triggered by the KG-upsert flow) or
 * manually via the "Verify graph" menu item. It polls the verification job and
 * renders a PASS/FAIL verdict + a per-strategy breakdown. It never blocks the
 * pipeline: the graph is already built by the time this runs, and any error is
 * shown inside the panel rather than thrown.
 */

import { startVerificationJob, getJobStatus } from "./pipeline_client.js";

const STRATEGY_BLURB = {
  S1: "Every source sentence is reflected in at least one triple.",
  S2: "Every node/edge is well-formed and uses a declared predicate.",
  S3: "No contradictory edges (e.g. Requires vs Prohibits) for a pair.",
  S4: "Every triple is structurally sound — no missing part or provenance.",
  S5: "No duplicate (subject, predicate, object) edges.",
};

let lastInput = { sentences: [], allowedPredicates: [] };
let polling = false;

const el = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function showVerificationPanel() {
  el("verification-panel")?.classList.remove("d-none");
}

export function hideVerificationPanel() {
  el("verification-panel")?.classList.add("d-none");
}

/** Hide + clear the verification panel (used when a new pipeline run starts). */
export function resetVerificationPanel() {
  lastInput = { sentences: [], allowedPredicates: [] };
  const panel = el("verification-panel");
  if (panel) panel.classList.add("d-none");
  const body = el("verification-body");
  if (body) body.innerHTML = "";
  const badge = el("verification-verdict-badge");
  if (badge) {
    badge.className = "verification-badge verification-badge-pending";
    badge.textContent = "…";
  }
}

/**
 * Wire the panel buttons + the "Verify graph" menu item. Idempotent.
 */
export function initVerificationPanel() {
  const reRun = el("verification-rerun-btn");
  if (reRun && !reRun.dataset.wired) {
    reRun.dataset.wired = "1";
    reRun.addEventListener("click", () => runVerification(lastInput));
  }
  const closeBtn = el("verification-close-btn");
  if (closeBtn && !closeBtn.dataset.wired) {
    closeBtn.dataset.wired = "1";
    closeBtn.addEventListener("click", () => hideVerificationPanel());
  }
  const menuBtn = el("verify-graph-menu-btn");
  if (menuBtn && !menuBtn.dataset.wired) {
    menuBtn.dataset.wired = "1";
    menuBtn.addEventListener("click", () => runVerification(lastInput));
  }
}

/**
 * Run verification and render the verdict.
 * @param {{sentences?: string[], allowedPredicates?: string[]}} input
 */
export async function runVerification({ sentences = [], allowedPredicates = [] } = {}) {
  if (polling) return; // avoid overlapping runs
  lastInput = { sentences, allowedPredicates };
  showVerificationPanel();
  renderLoading();
  polling = true;
  try {
    const start = await startVerificationJob({
      sentences,
      allowed_predicates: allowedPredicates,
    });
    const jobId = start.job_id;
    while (true) {
      const job = await getJobStatus(jobId);
      if (job.state === "done") {
        renderVerification(job.result?.verification);
        break;
      }
      if (job.state === "error") {
        renderError(job.error || "Verification failed.");
        break;
      }
      await sleep(1000);
    }
  } catch (e) {
    renderError(e?.message || String(e));
  } finally {
    polling = false;
  }
}

function renderLoading() {
  const badge = el("verification-verdict-badge");
  if (badge) {
    badge.className = "verification-badge verification-badge-pending";
    badge.textContent = "Verifying…";
  }
  const body = el("verification-body");
  if (body) {
    body.innerHTML =
      '<div class="text-muted small d-flex align-items-center gap-2 p-2">' +
      '<span class="spinner-border spinner-border-sm" role="status"></span>' +
      "Running the 5-strategy check over the graph…</div>";
  }
}

function renderError(message) {
  const badge = el("verification-verdict-badge");
  if (badge) {
    badge.className = "verification-badge verification-badge-fail";
    badge.textContent = "ERROR";
  }
  const body = el("verification-body");
  if (body) {
    body.innerHTML =
      '<div class="alert alert-danger py-2 small mb-0">Verification error: ' +
      escapeHtml(message) +
      "</div>";
  }
}

function renderVerification(verdict) {
  if (!verdict || !Array.isArray(verdict.strategies)) {
    renderError("Malformed verification result.");
    return;
  }

  const pass = verdict.verdict === "PASS";
  const badge = el("verification-verdict-badge");
  if (badge) {
    badge.className = `verification-badge ${pass ? "verification-badge-pass" : "verification-badge-fail"}`;
    badge.textContent = `${pass ? "PASS" : "FAIL"} · ${verdict.passed_count}/${verdict.total}`;
  }

  const rows = verdict.strategies
    .map((s) => renderStrategyRow(s))
    .join("");

  const body = el("verification-body");
  if (body) {
    body.innerHTML =
      `<div class="verification-meta text-muted small mb-1">Checked ${verdict.edge_count ?? "?"} triples in the graph.</div>` +
      `<div class="verification-rows">${rows}</div>`;
  }
}

function renderStrategyRow(s) {
  const skipped = !!s.skipped;
  const ok = !!s.passed;
  const icon = skipped ? "bi-dash-circle text-muted" : ok ? "bi-check-circle-fill text-success" : "bi-x-circle-fill text-danger";
  const scoreText =
    typeof s.score === "number" ? ` · ${(s.score * 100).toFixed(0)}%` : "";

  const flagged = Array.isArray(s.flagged) ? s.flagged : [];
  const hasMore = (s.flagged_total ?? flagged.length) > flagged.length;
  const detailsId = `verif-details-${s.id}`;

  const flaggedHtml = flagged.length
    ? `<ul class="verification-flagged">${flagged
        .map(
          (f) =>
            `<li><span class="verification-flagged-label">${escapeHtml(f.label)}</span>` +
            (f.reason ? `<span class="verification-flagged-reason"> — ${escapeHtml(f.reason)}</span>` : "") +
            "</li>"
        )
        .join("")}${hasMore ? `<li class="text-muted">…and ${(s.flagged_total - flagged.length)} more</li>` : ""}</ul>`
    : "";

  const expandable = flagged.length > 0;
  const errorHtml = s.error
    ? `<div class="verification-flagged-reason text-danger small">${escapeHtml(s.error)}</div>`
    : "";

  return `
    <div class="verification-row ${ok ? "" : skipped ? "" : "verification-row-fail"}">
      <button class="verification-row-head ${expandable ? "" : "verification-row-static"}" type="button"
              ${expandable ? `data-bs-toggle="collapse" data-bs-target="#${detailsId}"` : ""}>
        <i class="bi ${icon}"></i>
        <span class="verification-row-id">${escapeHtml(s.id)}</span>
        <span class="verification-row-name">${escapeHtml(s.name)}</span>
        <span class="verification-row-summary" title="${escapeHtml(STRATEGY_BLURB[s.id] || "")}">${escapeHtml(s.summary)}${scoreText}</span>
        ${expandable ? '<i class="bi bi-chevron-down verification-chevron"></i>' : ""}
      </button>
      ${errorHtml}
      ${expandable ? `<div id="${detailsId}" class="collapse">${flaggedHtml}</div>` : ""}
    </div>`;
}
