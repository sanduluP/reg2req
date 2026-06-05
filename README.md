<div align="center">

# 🧠 KBDebugger
### A Knowledge-Based Extractor for Trustworthy AI

*A human-in-the-loop pipeline that mines new knowledge from technical PDFs and grows a curated Neo4j knowledge graph of Trustworthy-AI concepts — one reviewed triple at a time.*

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Space-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/faris-abuali/kbdebugger-demo/)
[![Final Report](https://img.shields.io/badge/📄%20Final%20Report-PDF-red?style=for-the-badge)](https://github.com/Faris-Abuali/dfki-master-projects/blob/main/knowledge-based-extractor/Implementation_of_a_Knowledge_Based_Extractor_for_Trustworthy_AI.pdf)
[![Presentation](https://img.shields.io/badge/🎤%20Presentation-PPTX-orange?style=for-the-badge)](https://github.com/Faris-Abuali/dfki-master-projects/blob/main/knowledge-based-extractor/Implementation_of_a_Knowledge_Based_Extractor_for_Trustworthy_AI.pptx)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-4581C3?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Docling](https://img.shields.io/badge/Docling-PDF%20parsing-blueviolet)](https://github.com/DS4SD/docling)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

<br />

<a href="https://huggingface.co/spaces/faris-abuali/kbdebugger-demo/">
  <img src="https://img.shields.io/badge/▶️%20Try%20it%20live-on%20Hugging%20Face-fff?style=flat-square&labelColor=FFD21E" height="32" />
</a>

</div>

---

## ✨ Why this project?

Knowledge graphs that encode **Trustworthy-AI requirements** (e.g. *fairness*, *explainability*, *robustness*) age fast — new standards, papers, and guidelines appear weekly. Keeping such a graph fresh manually is slow and error-prone, while letting an LLM dump triples directly into the graph is risky.

**KBDebugger** sits *between* the two. It:

- 📥 ingests a new document (PDF or TXT),
- 🔎 finds only the parts that talk about a chosen concept,
- 🧪 decides — for every extracted statement — whether it is **EXISTING**, **PARTIALLY-NEW**, or **NEW** with respect to the current knowledge graph,
- 🖐️ asks a human to confirm before anything is written, and
- 🌱 upserts only the approved (Subject, Predicate, Object) triples into Neo4j.

The result is a **transparent, auditable, and incremental** way to grow a Trustworthy-AI knowledge base.

> 🎓 This repository is the implementation deliverable for the DFKI master project *“Implementation of a Knowledge-Based Extractor for Trustworthy AI”* (RPTU DSA). The full motivation, methodology, and evaluation live in the [📄 final report](https://github.com/Faris-Abuali/dfki-master-projects/blob/main/knowledge-based-extractor/Implementation_of_a_Knowledge_Based_Extractor_for_Trustworthy_AI.pdf) and the [🎤 presentation](https://github.com/Faris-Abuali/dfki-master-projects/blob/main/knowledge-based-extractor/Implementation_of_a_Knowledge_Based_Extractor_for_Trustworthy_AI.pptx).

---

## 🏛️ Pipeline Architecture (7 Stages)

<p align="center">
  <img src="https://github.com/Faris-Abuali/dfki-master-projects/raw/main/assets/knowledge-based-extractor/pipeline-architecture.png" alt="KBDebugger 7-stage pipeline architecture" width="90%" />
  <br />
  <sub><i>The 7-stage knowledge-extraction pipeline. The feedback loop lets newly integrated knowledge enrich the graph for future runs.</i></sub>
</p>

The pipeline is intentionally **modular** — each stage is a single function call in [`src/kbdebugger/pipeline/run.py`](src/kbdebugger/pipeline/run.py), so any block can be swapped or benchmarked in isolation.

| #  | Stage | Module | What it does |
|----|------|--------|-------------|
| 1️⃣ | **KG Subgraph Retrieval** | `graph/` (Neo4j) | Pull all relations around a user-chosen keyword |
| 2️⃣ | **Corpus → Qualities** | `extraction/` + `keyword_extraction/` ([Docling](https://github.com/DS4SD/docling), [KeyBERT](https://github.com/MaartenGr/KeyBERT), LLM Decomposer) | PDF → paragraphs → keyword gate → atomic *quality* sentences |
| 3️⃣ | **Vector Similarity Filter** | `subgraph_similarity/` (SentenceTransformers + exact NumPy vector search) | Keep only qualities related to the KG subgraph |
| 4️⃣ | **Novelty Comparator (LLM)** | `novelty/` | Classify each kept quality → `EXISTING` / `PARTIALLY_NEW` / `NEW` |
| 5️⃣ | **Triplet-First Human Oversight UI** | `ui/` (Flask + Cytoscape.js) | Auto-extract predicate-constrained triplets, then let the reviewer include / edit / reject |
| 6️⃣ | **Triplet Extraction (LLM)** | `extraction/triplet_extraction_batch.py` | Pull allowed-predicate (S, P, O) triples from reviewed qualities |
| 7️⃣ | **KG Upsert** | `graph/store.py` | Write approved triples back to Neo4j with provenance |

🧠 **LLM backends** are pluggable: hosted ([Groq](https://groq.com/), [OpenAI](https://openai.com/)) or local ([HuggingFace](https://huggingface.co/) Transformers).

---

## 🚀 Try it now (no install)

The simplest way to see the system in action is the hosted Space:

👉 **[huggingface.co/spaces/faris-abuali/kbdebugger-demo](https://huggingface.co/spaces/faris-abuali/kbdebugger-demo/)**

Upload a PDF, choose a keyword (e.g. `fairness`, `requirement`, `bias`), and watch the stages light up.

---

## 🛠️ Run it locally

### Prerequisites

- 🐍 **Python 3.10+**
- 🗄️ A reachable **Neo4j 5.x** instance (local Desktop, Docker, or Aura)
- 🔑 A **Groq** API key (or swap in your preferred LLM provider)
- 🐧 Linux / macOS (Windows works via WSL)

### 1. Clone

```bash
git clone https://github.com/islammesabah/KBExtraction.git
cd KBExtraction
```

### 2. Configure secrets

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_api_key
```

### 3. Install (one-time)

```bash
./setup.sh
```

This script:

- 🐍 creates a local virtualenv in `./venv/`,
- ⬆️ upgrades `pip` / `setuptools` / `wheel`,
- 📦 installs the pinned dependencies from `requirements.lock.txt` (CPU-only PyTorch is configured automatically),
- ✅ verifies the install.

### 4. Launch the app

```bash
./ui/run.sh
```

Then open 👉 **http://localhost:5002**

---

## 🐳 Docker / Hugging Face Spaces

The repo ships with a [`Dockerfile`](Dockerfile) tuned for **Hugging Face Spaces** (Gunicorn, port `7860`, CPU-only PyTorch, persistent HF cache):

```bash
docker build -t kbdebugger .
docker run --rm -p 7860:7860 --env-file .env kbdebugger
```

A redeploy of the public Space is one command away — see [`scripts/deploy_hf.sh`](scripts/deploy_hf.sh).

---

## ⚙️ Configuration knobs

All pipeline behavior is **environment-driven** (see [`src/kbdebugger/pipeline/config.py`](src/kbdebugger/pipeline/config.py)):

| Variable | Default | Meaning |
|---|---|---|
| `KB_RETRIEVAL_KEYWORD` | `requirement` | Topic to anchor the KG subgraph |
| `KB_LIMIT_PER_PATTERN` | `50` | Max relations per retrieval pattern |
| `KB_SOURCE_KIND` | `TEXT` | `TEXT` / `PDF_SENTENCES` / `PDF_CHUNKS` |
| `KB_PDF_PATH` | `data/SDS/InstructCIR.pdf` | Corpus PDF |
| `KB_ENCODER_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `KB_SIMILARITY_MODE` | `node_entity` | `node_entity` compares quality keyphrases to KG node labels; `sentence` keeps full quality sentence ↔ KG relation sentence comparison |
| `KB_ENTITY_EXTRACTION_MODE` | `keybert` | `keybert` reuses cached KeyBERT/SentenceTransformer models; `simple` uses regex chunks with scikit-learn stop words |
| `KB_ENTITY_KEYBERT_NGRAM_MAX` | `3` | Max phrase length for KeyBERT quality entity/keyphrase extraction |
| `KB_QUALITY_TO_KG_TOP_K` | `5` | Neighbors per candidate quality |
| `KB_MIN_SIMILARITY_THRESHOLD` | `0.55` | Cosine cutoff for vector filter |
| `KB_NODE_ENTITY_TOP_K` | `5` | Nearest KG nodes retrieved per quality entity/keyphrase |
| `KB_NODE_ENTITY_MAX_ENTITIES_PER_QUALITY` | `8` | Max candidate entities/keyphrases extracted per quality |
| `KB_KEYWORD_SYNONYMS_ENABLED` | `true` | Enable cached LLM/curated synonyms for keyword paragraph filtering |
| `KB_DECOMPOSER_PARALLEL` | `true` | Run decomposition batches concurrently |
| `KB_DECOMPOSER_MAX_WORKERS` | `2` | Conservative worker count for decomposition LLM calls |
| `KB_NOVELTY_PARALLEL` | `true` | Run novelty batches concurrently |
| `KB_NOVELTY_MAX_WORKERS` | `2` | Conservative worker count for novelty LLM calls |
| `KB_NOVELTY_LLM_TEMPERATURE` | `0.0` | Determinism for novelty decisions |
| `KB_TRIPLET_EXTRACTION_BATCH_SIZE` | `5` | Qualities per triplet-extraction call |
| `KB_TRIPLET_EXTRACTION_PARALLEL` | `true` | Run triplet extraction batches concurrently |
| `KB_TRIPLET_EXTRACTION_MAX_WORKERS` | `2` | Conservative worker count for triplet LLM calls |
| `DOCLING_ENABLE_OCR` | `false` | Toggle OCR in Docling |
| `DOCLING_ENABLE_TABLE_RECOGNITION` | `false` | Parse table structure |

### Current review flow

- The quality screen is audit-only: it shows novelty-reviewed qualities and provenance, but no manual selection checkboxes or quality-level export.
- Triplet extraction starts automatically for all visible reviewed qualities after novelty classification.
- `NEW` and `PARTIALLY_NEW` triplets are included for KG submission by default; `EXISTING` triplets remain visible for traceability and are excluded by default.
- No-fit qualities remain visible in the triplet review screen as skipped/warning rows, with original quality and source chunk available.
- The KG upsert route submits only rows where the reviewer left `include === true`.

### Predicate-constrained triplets

Triplet extraction is backend-owned and predicate-controlled. Allowed predicates live in [`src/kbdebugger/extraction/predicate_options.py`](src/kbdebugger/extraction/predicate_options.py).

- The extractor must use one of the allowed predicates exactly.
- If no allowed predicate fits, it returns `skipped_reason` instead of forcing a hallucinated relationship.
- `Fallback` is only valid for explicit fallback mechanisms; it is not a catch-all.

### Triplet review export

The triplet review Export button writes an `.xlsx` audit workbook with one row per non-deleted triplet. Rows are grouped by source chunk and include:

```text
Source Chunk
Original Quality
Nearest KG Match
Similarity Score
Extracted Triplet
Faithfulness (1-3)
Relevance (1-3)
Completeness (1-3)
```

Scoring headers include hidden rubric notes where SheetJS/Excel supports comments, and long review cells are configured for wrapped text where supported.

### Focused checks

```bash
PYTHONPATH=src venv/bin/python -m compileall \
  src/kbdebugger/keyword_extraction \
  src/kbdebugger/subgraph_similarity \
  src/kbdebugger/pipeline \
  ui/services

PYTHONPATH=src venv/bin/python -m pytest \
  tests/test_subgraph_similarity_modes.py \
  tests/test_post_docling_performance.py \
  tests/test_triplet_extraction_route.py \
  -q
```

---

## 🗂️ Repository layout

```text
KBExtraction/
├── 🧠 src/kbdebugger/        # Core Python package
│   ├── extraction/           # PDF/text parsing, decomposition, triplet extraction
│   ├── keyword_extraction/   # KeyBERT-based paragraph gate
│   ├── subgraph_similarity/  # Sentence-transformer index + similarity filter
│   ├── novelty/              # LLM novelty comparator
│   ├── graph/                # Neo4j store, retriever, Cytoscape exporters
│   ├── llm/                  # Groq / OpenAI / HuggingFace backends
│   ├── prompts/              # Versioned prompt templates
│   ├── human_oversight/      # Reviewer logging + decision API
│   ├── pipeline/             # Config + end-to-end runner
│   └── utils/                # Timing, JSON, progress, batching
├── 🖥️  ui/                    # Flask app (routes, services, templates, static)
├── 📚 docs/                  # Report drafts, design diagrams, references
├── 🗃️  data/                  # Seed corpus + Trustworthy-AI source PDFs
├── 🧰 scripts/               # Setup, deploy, SLURM jobs, wheel builders
├── 🧪 tools/                 # Standalone helpers (e.g. triplet importer)
├── 🛠️  configs/               # `config.ini` for legacy entry points
├── 🐳 Dockerfile             # HF Spaces / production image
├── ⚙️  setup.sh               # One-shot env bootstrap
└── 📦 requirements.lock.txt  # Reproducible deps
```

---

## 🔬 Glossary

We deliberately disambiguate graph terms:

- **Node** *(Neo4j; aka Vertex)* — an entity, e.g. `classification`, `supervised_learning`.
- **Relationship** *(Neo4j; aka Edge)* — a typed link, e.g. `(:Node)-[:IS_SUBCLASS_OF]->(:Node)`.
- **Triple / Triplet (S-P-O)** *(KG / NLP)* — `(Subject, Predicate, Object)` where **S** & **O** are Nodes and **P** is a Relationship type.

📝 *Convention:* extraction code uses **triple/predicate**; Neo4j code uses **relationship**.

---

## 📚 Read more

- 📄 **[Final Report (PDF)](https://github.com/Faris-Abuali/dfki-master-projects/blob/main/knowledge-based-extractor/Implementation_of_a_Knowledge_Based_Extractor_for_Trustworthy_AI.pdf)** — full methodology, evaluation, and discussion
- 🎤 **[Presentation (PPTX)](https://github.com/Faris-Abuali/dfki-master-projects/blob/main/knowledge-based-extractor/Implementation_of_a_Knowledge_Based_Extractor_for_Trustworthy_AI.pptx)** — defense slides
- 🤗 **[Live Demo Space](https://huggingface.co/spaces/faris-abuali/kbdebugger-demo/)** — try it without installing
- 🗂️ **[All DFKI master deliverables](https://github.com/Faris-Abuali/dfki-master-projects/tree/main/knowledge-based-extractor)**

---

## 🙏 Acknowledgements

Built during a master project at the [German Research Center for Artificial Intelligence (DFKI)](https://www.dfki.de/) in collaboration with [RPTU – Data Science & Artificial Intelligence](https://dsai.rptu.de/).

### 👨‍🏫 Supervisors

Huge thanks to my supervisors at the DFKI **Data Science & its Applications** group for their guidance, patience, and feedback throughout this project:

- 🧑‍🔬 **[Priyabanta Sandulu](https://dsa.dfki.de/team/members/priyabanta/)** — DFKI DSA
- 🧑‍🔬 **[Islam Mesabah](https://dsa.dfki.de/team/members/mesabah/)** — DFKI DSA

### 🧰 Powered by

- [🤗 HuggingFace Transformers & Sentence-Transformers](https://huggingface.co/)
- [🦆 Docling](https://github.com/DS4SD/docling) — structure-aware PDF parsing
- [🔑 KeyBERT](https://github.com/MaartenGr/KeyBERT) — keyword extraction
- [🦜🔗 LangChain](https://github.com/langchain-ai/langchain) — LLM orchestration
- [🗄️ Neo4j](https://neo4j.com/) — graph storage
- [⚡ Groq](https://groq.com/) — fast LLM inference
- [📐 Cytoscape.js](https://js.cytoscape.org/) — graph visualization in the browser

---
