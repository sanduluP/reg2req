<div align="center">

# reg2req
### Regulatory Text to Compliance Knowledge Graphs

*A human-in-the-loop pipeline that extracts structured knowledge from Trustworthy AI standards, detects cross-standard conflicts, and grows a curated Neo4j knowledge graph — one reviewed triple at a time.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-4581C3?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Docling](https://img.shields.io/badge/Docling-PDF%20parsing-blueviolet)](https://github.com/DS4SD/docling)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Why this project?

Trustworthy AI standards — EU AI Act, ISO/IEC 42001, ISO/IEC 5259-3 — express normative obligations in natural language using hedged modal constructions (*shall*, *should*, *may*). Compliance teams working across multiple co-regulating standards face thousands of assertions that nominally address the same concept but may impose different obligation strengths, conflicting scopes, or mutually inconsistent definitions.

**reg2req** addresses this with a seven-stage human-in-the-loop pipeline that:

- Conditions each extraction step on the **current graph state** — classifying candidates as `EXISTING`, `PARTIALLY_NEW`, or `NEW` — to avoid re-deriving known facts
- Records **deontic modality per triple** (`MANDATORY` / `RECOMMENDED` / `OPTIONAL` / `PROHIBITED`) making obligation-strength conflicts detectable
- Surfaces **inter-standard conflicts, normative gaps, and cross-standard concept coverage** through a structured comparison engine
- Keeps a human reviewer in the loop before anything is written to the graph

> This repository accompanies an AAAI 2027 submission. The EU AI Act Article 10 and ISO/IEC 42001 knowledge graphs produced by the system are released alongside the code to allow the community to fork and extend coverage to additional standards or organizational policy documents.

---

## Pipeline Architecture (7 Stages + Verifier)

| # | Stage | What it does |
|---|-------|-------------|
| 1 | **KG Subgraph Retrieval** | Pull relations around the user-chosen focus keyword from Neo4j |
| 2 | **Corpus → Quality Sentences** | PDF → paragraphs → keyword gate (KeyBERT) → synonym expansion → atomic quality sentences (LLM decomposer) |
| 3 | **Vector Similarity Filter** | Keep only qualities semantically related to the KG subgraph (SentenceTransformers + cosine threshold) |
| 4 | **Novelty Classification** | LLM classifies each quality as `EXISTING` / `PARTIALLY_NEW` / `NEW` relative to retrieved KG neighbours |
| 5 | **Triplet Extraction** | Extract `(Subject, Predicate, Object)` triples with deontic modality from quality sentences, constrained to the declared predicate vocabulary |
| 6 | **Human Review UI** | Reviewer includes, edits, or rejects each proposed triple; pre-merge KB preview shows conflicts and tensions before any write |
| 7 | **KG Upsert** | Write approved triples to Neo4j with append-only provenance; five-strategy post-upsert verifier (Coverage, Correctness, Consistency, Completeness, Minimality) |

The pipeline is modular — each stage is a single function call in [`src/kbdebugger/pipeline/run.py`](src/kbdebugger/pipeline/run.py).

---

## Quick Start

### Prerequisites

- **Python 3.11.9** (recommended — the lock file was generated for this version)
  The project declares `requires-python = ">=3.10"` but `requirements.lock.txt` pins exact
  package versions built for 3.11. Python 3.12/3.13 may work but some wheels might not
  resolve cleanly from the lock file.

  If you use **pyenv**, the `.python-version` file in the repo root will activate 3.11.9
  automatically:
  ```bash
  pyenv install 3.11.9   # one-time
  # .python-version takes effect automatically inside the repo directory
  ```

  If you do not use pyenv, create your venv with an explicit Python 3.11 binary:
  ```bash
  python3.11 -m venv venv
  ```

- A running **Neo4j 5.x** instance (local Desktop, Docker, or Aura)
- An **OpenAI-compatible LLM endpoint** — the DFKI internal server (`deepseek-r1:32b`), DeepSeek API, OpenAI, or any local model served via Ollama / vLLM

### 1. Clone

```bash
git clone https://github.com/sanduluP/reg2req.git
cd reg2req
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Neo4j connection
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# LLM backend — any OpenAI-compatible /v1/chat/completions endpoint
MODEL_BACKEND=http
MODEL_SERVICE_URL=https://api.deepseek.com/v1/chat/completions   # or your local server
MODEL_SERVICE_NAME=deepseek-chat
MODEL_API_KEY=your_api_key          # omit if your server does not require auth
REQUEST_TIMEOUT=120.0
```

### 3. Install

```bash
./setup.sh
```

This creates `./venv/`, installs pinned dependencies from `requirements.lock.txt` (CPU-only PyTorch configured automatically), and verifies the install.

### 4. Run

```bash
./ui/run.sh
```

Open **http://localhost:5002**

Alternatively, activate the venv and run directly:

```bash
source venv/bin/activate
PYTHONPATH=src python -m ui.app
```

### 5. Seed the knowledge graph (first run)

A fresh Neo4j is empty — every focus area returns an empty subgraph and the pipeline fails with *"No KG relations retrieved"*. Load the Trustworthy-AI baseline:

**UI:** toolbar → **Step 1 · Initialize graph → Load baseline knowledge**
(clears the whole graph and rebuilds from seed — a confirmation dialog is shown)

**CLI:**
```bash
source venv/bin/activate
PYTHONPATH=src python tools/seed_graph.py --reset   # clear + rebuild
PYTHONPATH=src python tools/seed_graph.py           # additive upsert (idempotent, no --reset)
```

The seed statements live in [`data/seed/trustworthy_ai_seed.txt`](data/seed/trustworthy_ai_seed.txt). Seeded edges carry `knowledge_type = "seed"` and `provenance_source = "seed:trustworthy-ai"` to distinguish ground truth from pipeline-extracted knowledge.

---

## Using a local LLM

The HTTP backend sends requests to any endpoint speaking the OpenAI `/v1/chat/completions` protocol. Set in `.env`:

```env
MODEL_BACKEND=http
MODEL_SERVICE_URL=http://localhost:11434/v1/chat/completions   # e.g. Ollama
MODEL_SERVICE_NAME=llama3.1
# MODEL_API_KEY=                                               # omit for local servers
```

Temperature is fixed at `0.0` for all classification and extraction calls to ensure deterministic output.

---

## Docker

```bash
docker build -t reg2req .
docker run --rm -p 5002:5002 --env-file .env reg2req
```

---

## Compare tab — cross-document analysis

After ingesting multiple standards, the **Compare** tab answers what the documents agree on, where they conflict, and where they leave obligations undefined:

- **Overlap & Coverage** — per-document contribution summary, assertions supported by ≥2 documents, and a concept × document matrix
- **Alignment** — proposes high-similarity `SAME_AS` pairs for review; accepted clusters merge concepts across documents
- **Conflicts** — typed candidates (modality conflict, definition divergence, taxonomy reversal, value conflict) adjudicated by an LLM judge (`AGREE` / `TENSION` / `CONTRADICT` + one-line rationale from verbatim source texts); confirmed findings written as `(:Conflict)` nodes
- **Ambiguity** — concepts obligated but never defined within the same document (normative gaps); hedge/vague-language usage per document

Each view exports to `.xlsx`. API under `/api/comparison/*` ([`ui/routes/comparison_routes.py`](ui/routes/comparison_routes.py)); analysis code in [`src/kbdebugger/comparison/`](src/kbdebugger/comparison/).

---

## Configuration reference

All pipeline behaviour is environment-driven ([`src/kbdebugger/pipeline/config.py`](src/kbdebugger/pipeline/config.py)). Put tuning values in your local `.env` (git-ignored).

| Variable | Default | Meaning |
|---|---|---|
| `KB_RETRIEVAL_KEYWORD` | `requirement` | Topic to anchor the KG subgraph |
| `KB_LIMIT_PER_PATTERN` | `50` | Max relations per retrieval pattern |
| `KB_SOURCE_KIND` | `TEXT` | `TEXT` / `PDF_SENTENCES` / `PDF_CHUNKS` |
| `KB_DROP_REFERENCE_SECTION` | `true` | Drop detected References/Bibliography sections before extraction |
| `KB_ENCODER_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (held constant across all evaluations) |
| `KB_SIMILARITY_MODE` | `node_entity` | `node_entity`: compare quality keyphrases to KG node labels; `sentence`: compare full quality sentences to KG relation sentences |
| `KB_QUALITY_TO_KG_TOP_K` | `5` | Neighbours per candidate quality for novelty classification |
| `KB_MIN_SIMILARITY_THRESHOLD` | `0.55` | Cosine cutoff for the vector similarity filter |
| `KB_NOVELTY_LLM_TEMPERATURE` | `0.0` | Deterministic novelty decisions |
| `KB_TRIPLET_EXTRACTION_BATCH_SIZE` | `5` | Qualities per triplet-extraction LLM call |
| `KB_SCHEMA_GROUNDING_ENABLED` | `true` | Use current Neo4j graph as schema context during triplet extraction |
| `KB_DECOMPOSER_MAX_WORKERS` | `2` | Concurrent LLM workers for decomposition |
| `KB_NOVELTY_MAX_WORKERS` | `2` | Concurrent LLM workers for novelty classification |
| `KB_TRIPLET_EXTRACTION_MAX_WORKERS` | `2` | Concurrent LLM workers for triplet extraction |
| `KB_KEYWORD_SYNONYMS_ENABLED` | `true` | Enable keyword synonym expansion via LLM |
| `KB_KEYWORD_SYNONYM_CACHE_ENABLED` | `true` | Read synonym cache before calling the LLM |
| `DOCLING_ENABLE_OCR` | `false` | Toggle OCR in Docling |

---

## Predicate vocabulary

Triplet extraction is constrained to a controlled predicate list in [`src/kbdebugger/extraction/predicate_options.py`](src/kbdebugger/extraction/predicate_options.py).

| Family | Predicates |
|--------|-----------|
| Normative | `Requires`, `Recommends`, `Permits`, `Prohibits`, `Defines`, `Provides`, `AppliesTo`, `Constrains`, `Measures` |
| Trustworthy AI | `IsDimensionOf`, `ContributesTo`, `MightMitigate`, `MightIntroduce`, `SurfacesRisk`, `IsThreatTo` |
| Data Science | `Implements`, `HasParameter`, `HasInput`, `HasOutput`, `Evaluates`, `Performs` |
| Taxonomy | `IsSubclassOf`, `IsEquivalentTo`, `IsA` |

If the LLM cannot fit a quality into any predicate, it returns a `skipped_reason` rather than hallucinating. Off-vocabulary predicates are tagged `Non-standard predicate`, excluded from submission by default, and only written if the reviewer explicitly includes them.

---

## Repository layout

```
reg2req/
├── src/kbdebugger/          # Core Python package
│   ├── extraction/          # PDF/text parsing, decomposition, triplet extraction
│   ├── keyword_extraction/  # KeyBERT-based paragraph gate
│   ├── subgraph_similarity/ # Sentence-transformer index + similarity filter
│   ├── novelty/             # LLM novelty comparator
│   ├── graph/               # Neo4j store, retriever, Cytoscape exporters
│   ├── llm/                 # HTTP / Groq / HuggingFace backends
│   ├── prompts/             # Versioned prompt templates
│   ├── comparison/          # Cross-document conflict, gap, overlap analysis
│   ├── pipeline/            # Config + end-to-end runner
│   └── utils/               # Timing, JSON, batching helpers
├── ui/                      # Flask app (routes, services, templates, static JS)
├── data/
│   └── seed/                # trustworthy_ai_seed.txt + trustworthy_ai_seed.json
├── scripts/                 # Setup, deploy, SLURM job scripts
├── tools/                   # seed_graph.py and other standalone helpers
├── Dockerfile               # Production image (port 5002)
├── setup.sh                 # One-shot environment bootstrap
└── requirements.lock.txt    # Pinned reproducible dependencies
```

---

## Running tests

```bash
source venv/bin/activate
PYTHONPATH=src python -m pytest tests/ -q
```

Focused checks:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_subgraph_similarity_modes.py \
  tests/test_post_docling_performance.py \
  tests/test_triplet_extraction_route.py \
  -q
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **Quality sentence** | An atomic statement produced by the LLM decomposer from which exactly one triple can be extracted |
| **Deontic modality** | Obligation strength: `MANDATORY` (shall), `RECOMMENDED` (should), `OPTIONAL` (may), `PROHIBITED` (shall not) |
| **Node** | A Neo4j entity, e.g. `data_quality`, `transparency` |
| **Triple / Triplet** | `(Subject, Predicate, Object)` where S & O are nodes and P is a relationship type |
| **Normative gap** | A concept that a standard obligates but never formally defines within the same document |

---

## Contributing

Contributions are welcome:

- **New standards** — ingest any normative document (ISO, NIST, organizational policy) by dropping a PDF into the UI
- **Extended seed knowledge** — edit [`data/seed/trustworthy_ai_seed.txt`](data/seed/trustworthy_ai_seed.txt) and open a PR
- **New predicate families** — extend [`src/kbdebugger/extraction/predicate_options.py`](src/kbdebugger/extraction/predicate_options.py)

Please open an issue before starting large changes.

---

## Acknowledgements

Developed at the [German Research Center for Artificial Intelligence (DFKI)](https://www.dfki.de/), Data Science & its Applications group, in collaboration with [RPTU — Data Science & Artificial Intelligence](https://dsai.rptu.de/).

Powered by [Docling](https://github.com/DS4SD/docling), [KeyBERT](https://github.com/MaartenGr/KeyBERT), [SentenceTransformers](https://www.sbert.net/), [Neo4j](https://neo4j.com/), and [Cytoscape.js](https://js.cytoscape.org/).
