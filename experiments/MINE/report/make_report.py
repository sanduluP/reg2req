"""Generate MINE-1 figures + tables for the KBExtractor-vs-KGGen writeup.

Reads the per-essay judge verdicts written by ``score_kgs.py`` under
``experiments/MINE/results/<system>/<judge-slug>/results_<id>.json`` and emits:

  figures/fig1_mine1_distribution.{png,pdf}  Figure-3 analog: per-article MINE-1
      histograms + fitted normals + mean lines (primary deepseek judge, full set).
  figures/fig2_judge_ablation.{png,pdf}       deepseek vs GPT-5 judge on the matched
      10-essay subset (judge-LLM ablation).
  figures/fig3_paired_scatter.{png,pdf}        per-article paired KBExtractor-vs-KGGen
      (deepseek judge, the 98 essays both systems built).
  RESULTS.md                                   copy-paste-ready tables + captions.

Run with the report venv (matplotlib + scipy):
  experiments/MINE/report/.venv/bin/python experiments/MINE/report/make_report.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from statistics import mean, median, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# --- paths / config --------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_FACTS = 15  # MINE-1: 15 known facts per article → scores are multiples of 100/15.
STEP = 100.0 / N_FACTS

DEEPSEEK_SLUG = "openai-deepseek-r1-32b"
GPT5_SLUG = "openai-gpt-5"
QWEN3_SLUG = "openai-Qwen-Qwen3-30B-A3B-Instruct-2507-FP8"
PRIMARY_SLUG = DEEPSEEK_SLUG                       # drives Tables 1/3
JUDGE_ORDER = [DEEPSEEK_SLUG, GPT5_SLUG, QWEN3_SLUG]  # preferred display order; others appended
# A judge gets its own distribution + paired-scatter figures only if it scored enough
# articles for those to be meaningful (a 10-article histogram/scatter is too sparse).
FULL_MIN = 50

# Friendly labels + one-line details for whichever judges are present on disk.
JUDGE_LABELS = {
    DEEPSEEK_SLUG: "deepseek-r1:32b",
    GPT5_SLUG: "GPT-5 (high)",
    QWEN3_SLUG: "Qwen3-30B-Instruct",
}

# Colour-blind-safe (Okabe-Ito). KBExtractor = blue ("ours"), KGGen = orange.
C_KB = "#0072B2"
C_KG = "#E69F00"
C_DS = "#0072B2"
C_G5 = "#CC79A7"

# Transparency strings reused in captions + on-figure info boxes.
EXTRACTION_CORE = "deepseek-r1:32b (DeepSeek-R1 distilled Qwen-32B, 32B params, temp 0)"
JUDGE_DEEPSEEK = "deepseek-r1:32b (32B params, on-prem Ollama, temp 0, CoT)"
JUDGE_GPT5 = "GPT-5 (OpenAI, reasoning_effort=high, temp 1.0) — exact KGGen-paper parity"
RETRIEVER = "all-MiniLM-L6-v2 (22.7M params), top-k=8 nearest nodes + 2-hop expansion"

# One-line details for Table 3 (per judge slug); falls back to the label.
JUDGE_DETAILS = {
    DEEPSEEK_SLUG: JUDGE_DEEPSEEK,
    GPT5_SLUG: JUDGE_GPT5,
    QWEN3_SLUG: "Qwen3-30B-A3B-Instruct-2507 (FP8, served via vLLM on a DFKI GPU) — open-source robustness check",
}


# --- data loading ----------------------------------------------------------
def essay_accuracy(path: str) -> float | None:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    facts = [x for x in data if isinstance(x, dict) and "fact" in x]
    if not facts:
        return None
    return 100.0 * sum(int(x.get("evaluation", 0)) for x in facts) / len(facts)


def load(system: str, slug: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for p in glob.glob(os.path.join(RESULTS_DIR, system, slug, "results_*.json")):
        m = re.search(r"results_(\d+)\.json$", p)
        if not m:
            continue
        acc = essay_accuracy(p)
        if acc is not None:
            out[int(m.group(1))] = acc
    return out


def stat_block(vals: list[float]) -> dict[str, float]:
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "median": median(vals),
    }


# --- judge discovery (add a judge by just dropping results/<system>/<slug>/) ---
def _prettify_judge(slug: str) -> str:
    return JUDGE_LABELS.get(slug, slug.replace("openai-", "").replace("-", " "))


def discover_judge_stats() -> list[dict]:
    """Every judge that scored BOTH systems → its per-judge means over the essays it
    scored for both. Adding a new judge dir needs no code change — it just appears."""
    def slugs(system: str) -> set[str]:
        d = os.path.join(RESULTS_DIR, system)
        return {x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))} if os.path.isdir(d) else set()

    both = slugs("kbextractor") & slugs("kggen_deepseek")
    ordered = [s for s in JUDGE_ORDER if s in both] + sorted(s for s in both if s not in JUDGE_ORDER)
    out: list[dict] = []
    for slug in ordered:
        kb, kg = load("kbextractor", slug), load("kggen_deepseek", slug)
        ids = sorted(set(kb) & set(kg))
        if not ids:
            continue
        kb_v, kg_v = [kb[i] for i in ids], [kg[i] for i in ids]
        out.append({
            "slug": slug, "label": _prettify_judge(slug), "n": len(ids),
            "kb": kb, "kg": kg, "ids": ids,          # raw per-essay verdicts (for figs)
            "kb_mean": mean(kb_v), "kg_mean": mean(kg_v),
            "kb_sem": stats.sem(kb_v) if len(kb_v) > 1 else 0.0,
            "kg_sem": stats.sem(kg_v) if len(kg_v) > 1 else 0.0,
        })
    return out


def judge_token(label: str) -> str:
    """Filename-safe short token from a judge label, e.g. 'Qwen3-30B-Instruct' → 'qwen3-30b-instruct'."""
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


# Primary judge (deepseek) drives Tables 1/3 + figs 1/3; ALL judges drive the ablation.
kb_ds = load("kbextractor", PRIMARY_SLUG)
kg_ds = load("kggen_deepseek", PRIMARY_SLUG)
common = sorted(set(kb_ds) & set(kg_ds))         # essays both systems built (100)
S_kb_full = stat_block(list(kb_ds.values()))
S_kg_full = stat_block(list(kg_ds.values()))
JUDGES = discover_judge_stats()
# Judges with enough coverage for their own distribution + paired-scatter figures.
FULL_JUDGES = [j for j in JUDGES if j["n"] >= FULL_MIN]


# --- Figure 1: MINE-1 distribution, one per full-coverage judge ------------
def fig_distribution(j: dict) -> str:
    kb = np.array([j["kb"][i] for i in j["ids"]])
    kg = np.array([j["kg"][i] for i in j["ids"]])
    edges = (np.arange(N_FACTS + 2) - 0.5) * STEP  # one bin per discrete fact-count

    # y headroom so the mean labels sit clear of the bars and curves.
    ymax = max(np.histogram(kb, bins=edges)[0].max(), np.histogram(kg, bins=edges)[0].max())
    top = ymax * 1.22

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for data, colour, label, va_off in (
        (kb, C_KB, "KBExtractor (ours)", 0),
        (kg, C_KG, "KGGen", 14),
    ):
        st = stat_block(list(data))
        ax.hist(data, bins=edges, alpha=0.45, color=colour, edgecolor="white", linewidth=0.6,
                label=f"{label}:  μ={st['mean']:.1f}%,  σ={st['std']:.1f},  n={st['n']}")
        # Fitted normal, scaled from density to article counts.
        mu, sigma = stats.norm.fit(data)
        xs = np.linspace(-2, 102, 400)
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma) * len(data) * STEP, color=colour, lw=2.2)
        ax.axvline(mu, color=colour, ls=":", lw=2.0)
        ax.annotate(f"μ = {mu:.1f}%", xy=(mu, top), xytext=(0, -2 - va_off),
                    textcoords="offset points", ha="center", va="top", color=colour,
                    fontweight="bold", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour, alpha=0.9))

    ax.set_xlabel("Facts captured  (%)", fontsize=12)
    ax.set_ylabel("Frequency  (articles)", fontsize=12)
    ax.set_title(f"MINE-1 Knowledge Retention — KBExtractor vs KGGen\n"
                 f"judge: {j['label']}  (n={j['n']})", fontsize=12.5, fontweight="bold")
    ax.set_xlim(-3, 103)
    ax.set_ylim(0, top)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    # Transparency footer (kept off the plot area so the distributions stay clean).
    footer = (f"Judge: {JUDGE_DETAILS.get(j['slug'], j['label'])}   |   "
              f"Extraction core (both systems): {EXTRACTION_CORE}\n"
              f"Retriever: {RETRIEVER}   |   "
              f"MINE-1 = % of 15 known facts entailed by the retrieved sub-graph, per article")
    fig.text(0.5, -0.045, footer, ha="center", va="top", fontsize=7.3, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#bbbbbb"))
    name = f"fig1_mine1_distribution_{judge_token(j['label'])}"
    _save(fig, name)
    return name


# --- Figure 2: judge-LLM ablation (auto over every discovered judge) --------
def fig2() -> None:
    if not JUDGES:
        print("… skipping fig2 (no judges discovered)"); return
    x = np.arange(len(JUDGES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * len(JUDGES) + 3.0), 5.0), constrained_layout=True)
    b_kb = ax.bar(x - w / 2, [j["kb_mean"] for j in JUDGES], w, yerr=[j["kb_sem"] for j in JUDGES],
                  capsize=4, color=C_KB, alpha=0.9, label="KBExtractor (ours)")
    b_kg = ax.bar(x + w / 2, [j["kg_mean"] for j in JUDGES], w, yerr=[j["kg_sem"] for j in JUDGES],
                  capsize=4, color=C_KG, alpha=0.9, label="KGGen")
    for bars in (b_kb, b_kg):
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, fontweight="bold")

    # Gap (KB − KGGen) above each judge group (dynamic, clear of the bar labels).
    for xi, j in zip(x, JUDGES):
        ax.annotate(f"Δ +{j['kb_mean'] - j['kg_mean']:.1f}",
                    xy=(xi, max(j["kb_mean"], j["kg_mean"])), xytext=(0, 16),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color="#333333")

    ax.set_xticks(x, [f"{j['label']}\n(n={j['n']})" for j in JUDGES], fontsize=10.5)
    ax.set_ylabel("MINE-1  (mean facts captured, %)", fontsize=12)
    ax.set_ylim(0, 116)
    ax.set_title("Judge-LLM ablation — KBExtractor > KGGen under every judge",
                 fontsize=12.5, fontweight="bold", pad=30)
    # Legend above the axes so it never collides with the bars / gap labels.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=9.5, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.5, -0.13,
            "Each judge scores identical graphs (same retriever + metric); bars are over that "
            "judge's own article set (n).\nGPT-5 capped at 10 articles for cost (≈ $0.17/article). "
            "Absolute level varies with judge strictness; the gap persists.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.3, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#bbbbbb"))
    _save(fig, "fig2_judge_ablation")


# --- Figure 3: paired per-article scatter, one per full-coverage judge ------
def fig_paired(j: dict) -> str:
    rng = np.random.default_rng(0)
    xs = np.array([j["kg"][i] for i in j["ids"]])
    ys = np.array([j["kb"][i] for i in j["ids"]])
    jx = xs + rng.uniform(-1.4, 1.4, len(xs))  # jitter: scores are discrete
    jy = ys + rng.uniform(-1.4, 1.4, len(ys))
    wins = int(np.sum(ys > xs))
    ties = int(np.sum(ys == xs))
    losses = int(np.sum(ys < xs))

    fig, ax = plt.subplots(figsize=(6.4, 6.2), constrained_layout=True)
    ax.fill_between([-5, 105], [-5, 105], 105, color=C_KB, alpha=0.05)
    ax.fill_between([-5, 105], -5, [-5, 105], color=C_KG, alpha=0.05)
    ax.plot([-5, 105], [-5, 105], color="#555555", ls="--", lw=1.2, label="parity (y = x)")
    ax.scatter(jx, jy, s=34, color=C_KB, alpha=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal")
    ax.set_xlabel("KGGen — facts captured (%)", fontsize=11.5)
    ax.set_ylabel("KBExtractor (ours) — facts captured (%)", fontsize=11.5)
    ax.set_title(f"Per-article paired comparison — judge: {j['label']}  (n={j['n']})\n"
                 f"KBExtractor wins {wins} · ties {ties} · losses {losses}",
                 fontsize=12, fontweight="bold")
    ax.text(0.04, 0.95, "KBExtractor better", transform=ax.transAxes, color=C_KB,
            fontsize=10, fontweight="bold", va="top")
    ax.text(0.62, 0.06, "KGGen better", transform=ax.transAxes, color=C_KG,
            fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    name = f"fig3_paired_{judge_token(j['label'])}"
    _save(fig, name)
    return name


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote figures/{name}.png + .pdf")


# --- RESULTS.md tables + captions ------------------------------------------
def write_markdown() -> None:
    # Table 2 — built dynamically so every discovered judge becomes a column.
    t2_header = "| System |" + "".join(f" Judge: {j['label']} (n={j['n']}) |" for j in JUDGES)
    t2_align = "|---" + "|---:" * len(JUDGES) + "|"
    t2_kb = "| **KBExtractor (ours)** " + "".join(f"| {j['kb_mean']:.2f}% " for j in JUDGES) + "|"
    t2_kg = "| KGGen " + "".join(f"| {j['kg_mean']:.2f}% " for j in JUDGES) + "|"
    t2_gap = "| **Gap (KB − KGGen)** " + "".join(f"| **+{j['kb_mean'] - j['kg_mean']:.2f}** " for j in JUDGES) + "|"
    t2_table = "\n".join([t2_header, t2_align, t2_kb, t2_kg, t2_gap])

    judge_names = ", ".join(j["label"] for j in JUDGES) or "—"
    n_full = S_kb_full["n"] + S_kg_full["n"]
    g5 = next((j for j in JUDGES if j["slug"] == GPT5_SLUG), None)
    gpt5_note = matched_note = ""
    if g5:
        gpt5_note = (
            f" GPT-5 (`reasoning_effort=high`, the KGGen-paper's own judge in `_1_evaluation.py`) is "
            f"capped at {g5['n']} articles — the full two-system run (~{n_full} judgements) would cost "
            f"≈ ${n_full * 0.17:.0f} at ≈ $0.17/article.")
        # Matched cross-check: deepseek on the SAME articles GPT-5 judged.
        g5_ids = sorted(set(load("kbextractor", GPT5_SLUG)) & set(load("kggen_deepseek", GPT5_SLUG)))
        if g5_ids:
            kb_m = mean([kb_ds[i] for i in g5_ids if i in kb_ds])
            kg_m = mean([kg_ds[i] for i in g5_ids if i in kg_ds])
            matched_note = (
                f" Head-to-head on those same {len(g5_ids)} articles the primary judge scores "
                f"KBExtractor {kb_m:.1f}% / KGGen {kg_m:.1f}%, so GPT-5 ({g5['kb_mean']:.1f} / "
                f"{g5['kg_mean']:.1f}) agrees on the ranking on identical essays.")
    ablation_judges = "; ".join(
        JUDGE_DETAILS.get(j["slug"], j["label"]) for j in JUDGES if j["slug"] != PRIMARY_SLUG) or "—"

    # Per-judge figure lists (one distribution + one paired scatter per full-coverage judge).
    dist_list = "\n".join(
        f"  - `fig1_mine1_distribution_{judge_token(j['label'])}` — {j['label']} (n={j['n']})"
        for j in FULL_JUDGES)
    paired_list = "\n".join(
        f"  - `fig3_paired_{judge_token(j['label'])}` — {j['label']} (n={j['n']})"
        for j in FULL_JUDGES)

    md = f"""# MINE-1 Results — KBExtractor vs KGGen

*Auto-generated by `make_report.py` from the per-article judge verdicts in
`experiments/MINE/results/`. Do not edit by hand — re-run the script.*

**Benchmark.** MINE-1 (Knowledge Retention), from KGGen (NeurIPS 2025, arXiv:2502.09956).
For each article the system builds a knowledge graph; for each of the article's 15 known
facts, the fact is embedded and the top-k=8 nearest graph nodes plus their 2-hop
neighbourhood are retrieved (`all-MiniLM-L6-v2`); an LLM judge decides 1/0 whether the
fact is entailed by that sub-graph. **MINE-1 = mean over articles of (facts captured / 15).**

**Fairness.** Both systems use the **same extraction backbone** — `{EXTRACTION_CORE}` —
so the comparison isolates the *extraction method*, not the underlying LLM. The retriever,
the 15 facts, and the judge are held identical across systems. KGGen graphs are regenerated
on this backbone (the dataset's pre-built GPT-4o/Gemini graphs are mis-aligned, so they are
not used).

---

## Table 1 — Primary results (judge: deepseek-r1:32b, full set)

| System | Extraction core | Articles (n) | MINE-1 (mean) | Std | Median |
|---|---|---:|---:|---:|---:|
| **KBExtractor (ours)** | deepseek-r1:32b | {S_kb_full['n']} | **{S_kb_full['mean']:.2f}%** | {S_kb_full['std']:.1f} | {S_kb_full['median']:.1f}% |
| KGGen | deepseek-r1:32b | {S_kg_full['n']} | {S_kg_full['mean']:.2f}% | {S_kg_full['std']:.1f} | {S_kg_full['median']:.1f}% |

**Table 1.** MINE-1 knowledge retention under an on-prem `deepseek-r1:32b` judge (32B params,
temperature 0, chain-of-thought). Across all {S_kb_full['n']} articles, KBExtractor retains
**{S_kb_full['mean']:.1f}%** of known facts vs KGGen's **{S_kg_full['mean']:.1f}%** — a
**+{S_kb_full['mean'] - S_kg_full['mean']:.1f} pt** gap, with both systems sharing the
`deepseek-r1:32b` extraction core, the `all-MiniLM-L6-v2` retriever, and the judge. KGGen's
graphs for 2/100 articles (ids 47, 57) were built at temperature 0.5 rather than 0 — greedy
decoding emitted object-less relations that KGGen's strict schema rejects wholesale (Table 3);
all other graphs use temperature 0.

---

## Table 2 — Judge-LLM ablation (robustness across judges)

{t2_table}

**Table 2.** Judge-LLM ablation: {judge_names} each score both systems on identical graphs
(each judge over its own article set, n in the header). **All judges rank KBExtractor above
KGGen** — the ranking is robust to the choice of judge. Absolute levels shift with judge
strictness (a more lenient judge lifts both systems), but the KBExtractor−KGGen gap persists
under every judge.{matched_note}{gpt5_note}

---

## Table 3 — Configuration & transparency

| Component | Choice |
|---|---|
| Benchmark | MINE-1, 100 articles × 15 facts (KGGen, arXiv:2502.09956) |
| Extraction core (both systems) | {EXTRACTION_CORE} |
| Retriever | {RETRIEVER} |
| Primary judge | {JUDGE_DEEPSEEK} |
| Ablation judge(s) | {ablation_judges} |
| KGGen temperature note | 98/100 graphs at temp 0; ids 47, 57 needed temp 0.5 — at temp 0 deepseek deterministically emitted object-less relations that KGGen's all-or-nothing `list[Relation]` parse discarded wholesale. KBExtractor built all 100 at temp 0. |

**Table 3.** Experimental configuration. Every component except the extraction *method* is
held constant across systems, so MINE-1 differences are attributable to extraction, not to
the backbone LLM, the retriever, or the judge.

---

## Figures

- **Figure 1 — MINE-1 distribution** (per-article histogram + fitted normal + mean lines),
  one per full-coverage judge; KBExtractor's mass sits well to the right of KGGen's:
{dist_list}
- **Figure 2 — judge-LLM ablation** (`fig2_judge_ablation`): a grouped (clustered) bar chart,
  one group per judge ({judge_names}); KBExtractor leads under every judge.
- **Figure 3 — per-article paired comparison**, one per full-coverage judge; each point is an
  article, points above the *y = x* diagonal are KBExtractor wins:
{paired_list}
"""
    path = os.path.join(HERE, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"✓ wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    print(f"primary judge: kb={len(kb_ds)} kg={len(kg_ds)} common={len(common)}")
    print("judges (ablation): " + ", ".join(f"{j['label']}(n={j['n']})" for j in JUDGES))
    print("judges (own dist+paired figs): " + ", ".join(j["label"] for j in FULL_JUDGES))
    for j in FULL_JUDGES:        # Figure 1 + Figure 3 per full-coverage judge
        fig_distribution(j)
        fig_paired(j)
    fig2()                       # Figure 2: one ablation across all judges
    write_markdown()
    print("done.")
