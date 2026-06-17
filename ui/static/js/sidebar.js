import { formatPredicateLabel } from "./utils/predicate_format.js";

const detailsPanel = document.getElementById("details-panel") // aka sidebar or insights panel

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setDetailsTitle(title) {
  // Change the title of the details panel (e.g. to show "Node details" or "Edge details")
  const el = document.getElementById("details-title");
  if (el) el.textContent = title;
}

function copyBtn(value) {
  if (!value) return "";
  return `
    <button class="btn btn-sm py-0 px-2 copy-btn"
            type="button"
            data-copy="${escapeHtml(value)}"
            title="Copy to clipboard">
      <i class="bi bi-copy"></i>
    </button>
  `;
}

function wireCopyButtons(container) {
  if (!container || typeof container.querySelectorAll !== "function") {
    console.warn("[panel] wireCopyButtons called with non-element:", container);
    return;
  }

  container.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      const text = btn.getAttribute("data-copy") || "";
      try {
        await navigator.clipboard.writeText(text);
        // tiny feedback
        const old = btn.innerHTML;
        // change to bootstrap check icon
        // btn.innerHTML = `<i class="bi bi-clipboard-check-fill"></i>`;
        btn.innerHTML = `<i class="bi bi-check2"></i>`;
        // btn.textContent = "✅";
        setTimeout(() => (btn.innerHTML = old), 700);
      } catch (err) {
        console.error("Clipboard failed:", err);
      }
    });
  });
}

/**
 * Parse ISO timestamp safely.
 * Accepts:
 * - "2026-02-11T17:09:09.801354+00:00"
 * - "2026-02-11T17:09:09+00:00"
 * Returns Date or null
 */
function parseIso(s) {
  if (!s) return null;
  const d = new Date(s);
  return isNaN(d.getTime()) ? null : d;
}

/**
 * Facebook-ish relative time:
 * - just now
 * - 5 minutes ago
 * - 2 hours ago
 * - yesterday
 * - 3 days ago
 * - 2 weeks ago
 * - 3 months ago
 * - 2 years ago
 */
function timeAgo(isoString) {
  const d = parseIso(isoString);
  if (!d) return null;

  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const future = diffMs < 0;
  const diff = Math.abs(diffMs);

  const sec = Math.floor(diff / 1000);
  const min = Math.floor(sec / 60);
  const hr = Math.floor(min / 60);
  const day = Math.floor(hr / 24);
  const week = Math.floor(day / 7);
  const month = Math.floor(day / 30);
  const year = Math.floor(day / 365);

  const suffix = future ? "from now" : "ago";

  if (sec < 10) return future ? "in a few seconds" : "just now";
  if (sec < 60) return `${sec} seconds ${suffix}`;
  if (min < 60) return `${min} minute${min === 1 ? "" : "s"} ${suffix}`;
  if (hr < 24) return `${hr} hour${hr === 1 ? "" : "s"} ${suffix}`;
  if (day === 1) return future ? "tomorrow" : "yesterday";
  if (day < 7) return `${day} days ${suffix}`;
  if (week < 5) return `${week} week${week === 1 ? "" : "s"} ${suffix}`;
  if (month < 12) return `${month} month${month === 1 ? "" : "s"} ${suffix}`;
  return `${year} year${year === 1 ? "" : "s"} ${suffix}`;
}

function formatAbsolute(isoString) {
  const d = parseIso(isoString);
  if (!d) return null;
  // Example: Feb 11, 2026, 17:09 UTC
  // (Uses user's locale but stable-ish format)
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

function timeRow({ icon = null, emoji = null, label, isoString }) {
  if (!isoString) return "";

  const abs = formatAbsolute(isoString) || isoString;
  const rel = timeAgo(isoString);

  const iconPart = icon ? `<i class="bi ${escapeHtml(icon)} me-1"></i>` : (emoji ? `${escapeHtml(emoji)} ` : "");

  return `
    <div class="mb-3">
      <div class="text-muted small">${iconPart}${escapeHtml(label)}</div>
      <div class="d-flex align-items-start justify-content-between gap-2">
        <div style="min-width:0;">
          <div class="fw-semibold" title="${escapeHtml(isoString)}" style="word-break: break-word;">
            ${escapeHtml(abs)}
          </div>
          ${rel ? `<div class="text-muted small">(${escapeHtml(rel)})</div>` : ""}
        </div>
        ${copyBtn(abs)}
      </div>
    </div>
  `;
}

function infoRow({ icon = null, emoji = null, label, value, title = "" }) {
  if (value == null || value === "") return "";

  const iconPart = icon
    ? `<i class="bi ${escapeHtml(icon)} me-1"></i>`
    : (emoji ? `${escapeHtml(emoji)} ` : "");

  return `
    <div class="mb-3">
      <div class="text-muted small">${iconPart}${escapeHtml(label)}</div>
      <div class="d-flex align-items-start justify-content-between gap-2">
        <div class="fw-semibold" style="word-break: break-word; min-width:0;" title="${escapeHtml(title)}">
          ${escapeHtml(value)}
        </div>
        ${copyBtn(value)}
      </div>
    </div>
  `;
}


function sentenceCallout(sentence) {
  if (!sentence) return "";

  return `
    <div class="p-3 rounded-3 bg-light border mb-3">
      <div class="text-muted small mb-1 d-flex justify-content-between">
        <span>
          <i class="bi bi-quote me-1"></i>
          Sentence
        </span>
        ${copyBtn(sentence)}
      </div>
      <div class="fw-semibold" style="line-height: 1.4;">${escapeHtml(sentence)}</div>
    </div>
  `;
}

function metadataAccordion(idSuffix, jsonObj) {
  const jsonText = JSON.stringify(jsonObj ?? {}, null, 2);
  const accId = `meta-acc-${idSuffix}`;
  const headingId = `meta-heading-${idSuffix}`;
  const collapseId = `meta-collapse-${idSuffix}`;

  return `
    <div class="accordion mt-3" id="${accId}">
      <div class="accordion-item">
        <h2 class="accordion-header" id="${headingId}">
          <button class="accordion-button collapsed py-2" type="button"
                  data-bs-toggle="collapse" data-bs-target="#${collapseId}"
                  aria-expanded="false" aria-controls="${collapseId}">
            <i class="bi bi-list-columns me-1"></i>Relationship metadata
          </button>
        </h2>
        <div id="${collapseId}" class="accordion-collapse collapse"
             aria-labelledby="${headingId}" data-bs-parent="#${accId}">
          <div class="accordion-body">
            <pre class="bg-white border rounded p-2 small mb-0" style="white-space: pre; overflow:auto;">${escapeHtml(jsonText)}</pre>
          </div>
        </div>
      </div>
    </div>
  `;
}


function badgeTriplet(src, rel, tgt) {
  return `
    <div class="d-flex flex-wrap align-items-center gap-1">
      <span class="badge text-bg-secondary">${escapeHtml(src)}</span>
      <span class="text-muted">→</span>
      <span class="badge text-bg-primary">${escapeHtml(rel)}</span>
      <span class="text-muted">→</span>
      <span class="badge text-bg-secondary">${escapeHtml(tgt)}</span>
    </div>
  `;
}

/**
 * A horizontally-scrollable sentence line (only scrolls if needed).
 * Keeps layout clean without ugly truncation.
 */
function scrollableSentenceLine(sentence) {
  if (!sentence) return "";
  return `
    <div class="text-muted small mt-2"
         style="overflow-x:auto; white-space:nowrap; -webkit-overflow-scrolling: touch;">
      ${escapeHtml(sentence)}
    </div>
  `;
}

export function renderEmptyDetails(container) {
  // showSidebar();
  setDetailsTitle("Concept Insights");

  container.innerHTML = `
    <p class="placeholder-text">Select a node or an edge to view details.</p>
  `;

  hideSidebar();
}


export function renderEdgeDetails(container, edgeData, endpoints = {}) {
  showSidebar();
  setDetailsTitle("Relationship details");
  const props = edgeData?.properties || {};

  const relationLabel = edgeData?.label || props?.label || "relation";
  const relationDisplayLabel = formatPredicateLabel(relationLabel);
  const sentence = props.sentence || "";
  const createdAt = props.created_at || null;
  const updatedAt = props.last_updated_at || null;

  // Source/target node names come from the graph topology (the relationship
  // endpoints), not from stored redundant properties. They are shown only in
  // the metadata view, never as a prominent parent row.
  const sourceName = endpoints.source || props.source || "";
  const targetName = endpoints.target || props.target || "";
  const metadata = { source: sourceName, target: targetName, ...props };

  container.innerHTML = `
    <div class="mb-2">
      <span class="badge text-bg-primary" title="${escapeHtml(relationLabel)}">${escapeHtml(relationDisplayLabel)}</span>
    </div>

    ${sentenceCallout(sentence)}

    ${timeRow({
    icon: "bi-clock-history",
    label: "Created",
    isoString: createdAt
  })}
    ${timeRow({
    icon: "bi-arrow-repeat",
    label: "Last updated",
    isoString: updatedAt
  })}

    ${metadataAccordion(edgeData?.id || "edge", metadata)}
  `;
  wireCopyButtons(container);
}


export function renderNodeDetails(container, nodeData, incidentEdges, onEdgePick) {
  showSidebar();
  setDetailsTitle("Node details");

  const label = nodeData?.label ?? "node";
  const id = nodeData?.id ?? "";

  // Make the whole card clickable (no button)
  const items = incidentEdges.map((e) => {
    const d = e.data();

    const srcLabel = e.source()?.data("label") ?? d.source;
    const tgtLabel = e.target()?.data("label") ?? d.target;
    const relLabel = d.label ?? "relation";
    const relDisplayLabel = formatPredicateLabel(relLabel);
    const sentence = (d.properties && d.properties.sentence) ? d.properties.sentence : "";

    return `
      <div class="border rounded p-2 mb-2 relation-card"
           role="button"
           tabindex="0"
           data-edge-id="${escapeHtml(d.id)}"
           style="cursor:pointer;"
           title="↗️ Go to relation">
        ${badgeTriplet(srcLabel, relDisplayLabel, tgtLabel)}
        ${scrollableSentenceLine(sentence)}
      </div>
    `;
  }).join("");

  container.innerHTML = `
    <div class="mb-2">
      <span class="badge text-bg-secondary">Node</span>
    </div>

    ${infoRow({ label: "Label", value: label })}
    <div class="text-muted small">ID: ${id}</div>

    <hr class="my-3"/>

    <div class="fw-semibold mb-2">🔗 Relations (${incidentEdges.length})</div>
    ${incidentEdges.length ? items : `<div class="text-muted small">No relations found for this node.</div>`}
  `;

  // Click + keyboard access
  container.querySelectorAll(".relation-card").forEach((card) => {
    const edgeId = card.getAttribute("data-edge-id");

    const pick = () => onEdgePick(edgeId);

    card.addEventListener("click", pick);
    card.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        pick();
      }
    });
  });

  wireCopyButtons(container);
}

export function hideSidebar() {
  detailsPanel.classList.add("d-none");
}

export function showSidebar() {
  detailsPanel.classList.remove("d-none");
}

const closeBtn = document.getElementById("sidebar-close-btn");

if (closeBtn) {
  closeBtn.addEventListener("click", () => {
    hideSidebar();
  });
}

