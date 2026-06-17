/**
 * Seed controller
 * ---------------
 * Wires the "Seed knowledge graph (sample)" action: one click seeds Neo4j
 * with curated Trustworthy-AI ground truth so a fresh database is not empty.
 *
 * Seeding runs as a background job (POST /api/graph/seed); we poll the shared
 * pipeline job endpoint, then re-render the current keyword's subgraph if one
 * is selected.
 */

import { seedGraph } from "./graph_client.js";
import { getJobStatus } from "./pipeline_client.js";
import { showGlobalLoading, hideGlobalLoading } from "./modals/global_loading_modal.js";
import { showToast } from "./toast.js";
import { refreshGraphForKeyword } from "./graph_refresh.js";
import { confirmModal } from "./modals/confirm_modal.js";

function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

/**
 * @param {Object} params
 * @param {string} params.seedBtnId
 * @param {string} [params.keywordSelectId="keyword-select"]
 */
export function wireSeedButton({ seedBtnId, keywordSelectId = "keyword-select" }) {
  const seedBtn = document.getElementById(seedBtnId);
  if (!seedBtn || seedBtn.dataset.wired) return;
  seedBtn.dataset.wired = "1";

  seedBtn.addEventListener("click", async () => {
    // Initialize clears the database, then rebuilds the curated baseline.
    const ok = await confirmModal({
      title: "Initialize knowledge graph?",
      body:
        "This clears the entire Neo4j graph (all nodes and relationships) and rebuilds it " +
        "from the curated Trustworthy-AI baseline. This cannot be undone.",
      confirmText: "Clear & initialize",
      cancelText: "Cancel",
      confirmBtnClass: "btn-primary",
    });
    if (!ok) return;

    seedBtn.disabled = true;
    showGlobalLoading(
      "Initializing knowledge graph…",
      "Clearing the database and creating the curated baseline in Neo4j."
    );

    try {
      const start = await seedGraph({ reset: true });
      const jobId = start.job_id;

      while (true) {
        const job = await getJobStatus(jobId);
        if (job.state === "done") break;
        if (job.state === "error") {
          throw new Error(job.error || "Seeding failed.");
        }
        await sleep(1000);
      }

      const job = await getJobStatus(jobId);
      const summary = job.result?.seed || {};

      showToast({
        type: "success",
        title: "🌱 Knowledge graph initialized",
        message: `${summary.relations ?? 0} relationships · ${summary.nodes ?? 0} concepts. Pick a focus area to explore.`,
      });

      // If a keyword is already selected, re-render its subgraph now.
      const select = document.getElementById(keywordSelectId);
      const keyword = (select?.value || "").trim();
      if (keyword) {
        await refreshGraphForKeyword(keyword);
      }
    } catch (e) {
      showToast({
        type: "error",
        title: "❌ Seeding failed",
        message:
          (e.message || String(e)) +
          " — check NEO4J_URI / credentials in .env and that the database is reachable.",
      });
    } finally {
      hideGlobalLoading();
      seedBtn.disabled = false;
    }
  });
}
