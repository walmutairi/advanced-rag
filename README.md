# Advanced RAG — Question-Bank-First Retrieval

A fully local RAG system that mines an **advanced question bank** out of your PDFs
up front, then answers by searching those questions first and falling back to
classical vector retrieval — refusing outright when neither has the answer.
You can **edit and add questions** in the bank at any time; those curated entries
act as **guardrails**, steering answers toward approved phrasing and grounding
before the system ever reaches for raw chunk retrieval.

Nothing leaves your machine. All inference runs on [Ollama](https://ollama.com).

This is a **local, single-user** application: there is no login, no per-user
isolation, and every browser session attached to the same Streamlit process shares
one Chroma index and upload directory. That is appropriate for a personal
workstation; if you ever deploy it on a shared server, put it behind a reverse
proxy with authentication and run one instance per tenant.

User-supplied PDF text, LLM answers, and manually entered question-bank entries
are rendered with HTML escaping in the UI so a malicious document cannot inject
script into the page. Treat uploaded PDFs as untrusted input anyway.

---

## The core idea

Most RAG systems embed chunks and hope the user's question lands near the right
one. That fails whenever the question and the passage share meaning but not
vocabulary.

This system inverts it. At ingest time an LLM reads every chunk and writes the
questions that passage actually answers — along with grounded answers, verbatim
evidence quotes, and paraphrases of how a real person might phrase the same
question. At query time we match **question against question**, which is a far
easier semantic comparison than question-against-prose.

Vector search over the raw chunks is still there, but as a *fallback* rather than
the primary path.

## Retrieval tiers

```
          user query
              │
              ▼
   ┌──────────────────────────┐
   │ TIER 1 — Question bank   │   hybrid dense + BM25 over extracted
   │ hit ≥ 0.72               │   questions AND their paraphrases,
   └──────────┬───────────────┘   then an LLM relevance gate
              │ miss
              ▼
   ┌──────────────────────────┐
   │ TIER 2 — Classical RAG   │   query expansion + RRF over
   │ hit ≥ 0.45               │   document chunks in Chroma
   └──────────┬───────────────┘
              │ miss
              ▼
   ┌──────────────────────────┐
   │ TIER 3 — Refuse          │   "I could not find the answer to your
   │                          │    question in the provided documents."
   └──────────────────────────┘
```

**Tier 1 returns *all* matches within `QUESTION_MULTI_MARGIN` of the best one.**
When a question genuinely has several answers across the corpus, every one is
passed to the LLM, which synthesises across them and surfaces disagreement
explicitly rather than silently picking a winner.

## The four hallucination guards

The requirement that the model never answer from its own knowledge is enforced in
four independent places, so no single failure opens the gate:

1. **Evidence grounding at extraction.** Every Q/A pair must quote its source
   verbatim. Quotes are checked against the chunk text; a pair whose evidence
   cannot be located is discarded before it ever reaches the index.
2. **Similarity thresholds.** Below `VECTOR_MATCH_THRESHOLD` nothing is retrieved
   and no generation call is made at all — the refusal path never touches the LLM.
3. **The `INSUFFICIENT_CONTEXT` sentinel.** The answering prompt forbids outside
   knowledge and instructs the model to emit a sentinel when the sources fall
   short. The sentinel is converted into a refusal.
4. **Post-hoc groundedness verification.** A second LLM pass decomposes the draft
   answer into atomic claims and checks each against the retrieved context. Score
   below `GROUNDEDNESS_MIN_SCORE` and the draft is thrown away — a partly
   fabricated answer is worse than none.

## What makes the question bank "advanced"

Extraction is prompted for coverage across ten cognitive levels, not just
definitions:

| | | |
|---|---|---|
| `factual` | `definitional` | `conceptual` |
| `causal` | `comparative` | `quantitative` |
| `procedural` | `multi_hop` | `application` |
| `critical` | | |

Each chunk is pushed to yield at least one genuinely hard pair — `multi_hop`
(requires joining two separate statements), `application` (apply the material to a
novel scenario), or `critical` (limitations, assumptions, implications) — wherever
the material supports it. Every pair also carries a difficulty rating, keywords,
and paraphrases indexed as separate retrieval rows.

---

## Setup

```bash
ollama serve                    # if not already running
ollama pull gemma4:31b          # answering + extraction
ollama pull bge-m3              # embeddings (1024-dim, multilingual)

python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Run

```bash
./.venv/bin/streamlit run app.py
```

Upload PDFs in the sidebar, wait for extraction, then search.

### Headless CLI

```bash
./.venv/bin/python -m rag.pipeline ingest paper.pdf notes.pdf
./.venv/bin/python -m rag.pipeline ask "what were the reported limitations?"
./.venv/bin/python -m rag.pipeline stats
```

### Eval harness

Edit [eval/sample_questions.json](eval/sample_questions.json) with real questions from your
corpus (aim for 50–200 for a serious quality bar), then:

```bash
./.venv/bin/python -m eval.eval_retrieval
./.venv/bin/python -m eval.eval_retrieval --route-only
./.venv/bin/python -m eval.eval_retrieval --baseline   # vs naive vector RAG
```

Successful ingests also run a fast route-only eval automatically when
`RAG_ENABLE_POST_INGEST_EVAL=1`.

### HTTP API

```bash
./.venv/bin/python -m rag.api
curl -s http://127.0.0.1:8765/health
curl -s -X POST http://127.0.0.1:8765/ask \
  -H 'content-type: application/json' \
  -d '{"query":"What are the limitations?"}'
```

---

## A note on ingest time

Question extraction runs `gemma4:31b` over **every chunk**, which is the expensive
part by a wide margin — budget roughly 20-40 seconds per chunk. A 100-page PDF is
plausibly an hour. This was a deliberate choice: the question bank is built once
and queried many times, so quality there pays off on every subsequent search.

To trade some of that quality for a large speed gain, point extraction at a
smaller model while keeping a larger one for answering:

```bash
RAG_EXTRACT_MODEL=gemma3:4b RAG_ANSWER_MODEL=gemma4:31b \
  ./.venv/bin/python -m rag.pipeline ingest paper.pdf
```

Ingest is idempotent — documents are keyed by a content hash, so re-uploading the
same file is skipped rather than duplicated.

---

## Configuration

Everything tunable lives in [config.py](config.py), and every value can be
overridden by environment variable.

| Setting | Default | Effect |
|---|---|---|
| `RAG_LLM_MODEL` | `gemma4:31b` | Legacy alias for both extract and answer when the vars below are unset |
| `RAG_EXTRACT_MODEL` | *(same as `RAG_LLM_MODEL`)* | Ingest-time question extraction |
| `RAG_ANSWER_MODEL` | *(same as `RAG_LLM_MODEL`)* | Query-time reranking and answer synthesis |
| `RAG_EMBED_MODEL` | `bge-m3:latest` | Embeddings for both collections |
| `RAG_QUESTION_MATCH_THRESHOLD` | `0.72` | Tier 1 bar. Raise for precision, lower for recall |
| `RAG_QUESTION_MULTI_MARGIN` | `0.08` | How far below the best hit still counts as an answer |
| `RAG_VECTOR_MATCH_THRESHOLD` | `0.45` | Tier 2 bar. **Lower this and you weaken the refusal guarantee** |
| `RAG_HYBRID_DENSE_WEIGHT` | `0.65` | Dense vs. BM25 blend. `1.0` = pure embeddings |
| `RAG_QA_PER_CHUNK_MAX` | `7` | Question bank density per chunk |
| `RAG_ENABLE_GROUNDEDNESS_CHECK` | `1` | Post-hoc verification. Disabling doubles speed and removes guard #4 |
| `RAG_GROUNDEDNESS_MIN_SCORE` | `1.0` | Fraction of claims that must be supported |
| `RAG_SKIP_GROUNDEDNESS_ON_QUESTION_BANK` | `1` | Skip audit for strong Tier-1 hits |
| `RAG_ENABLE_HYBRID_ROUTE` | `1` | Merge Tier-2 chunks when Tier-1 is weak-but-accepted |
| `RAG_EXTRACT_NEIGHBOR_CHUNKS` | `1` | Give extract model prev/next chunk context |
| `RAG_ENABLE_OCR_CACHE` | `1` | Persist OCR page text under `data/cache/ocr` |
| `RAG_ENABLE_EXTRACT_CACHE` | `1` | Skip re-extracting unchanged chunk text |
| `RAG_EMBED_BATCH_SIZE` | `32` | Texts per Ollama embed request |

### Tuning the refusal rate

If the system refuses too often, the question bank is probably too sparse — raise
`RAG_QA_PER_CHUNK_MAX` and re-ingest before you touch the thresholds. Lowering
`RAG_VECTOR_MATCH_THRESHOLD` is the one change that directly trades away the
no-hallucination guarantee, so make it last and deliberately.

---

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Advanced RAG system architecture diagram" width="900"/>
</p>

The diagram above shows the full pipeline: three entry points feed a single orchestrator, which runs either the **ingest path** (build the index once) or the **query path** (answer many times). Manual question-bank edits act as guardrails throughout.

### Entry points

| Part | Module | What it does |
|---|---|---|
| **Streamlit UI** | [app.py](app.py) | Upload PDFs, watch ingest progress, browse and edit the question bank, and chat with citations. |
| **HTTP API** | [rag/api.py](rag/api.py) | Headless `/ask` and `/health` endpoints for scripts or other apps on the same machine. |
| **CLI** | [rag/pipeline.py](rag/pipeline.py) | `ingest`, `ask`, and `stats` commands for terminal-only workflows. |

All three call the same **RAG Pipeline** orchestrator ([rag/pipeline.py](rag/pipeline.py)), which wires together loading, extraction, indexing, retrieval, and answering.

---

### Ingest path (build once)

Turns uploaded PDFs into a searchable question bank plus a chunk index.

| # | Part | Module | What it does |
|---|---|---|---|
| 1 | **PDF upload** | `data/uploads` | Raw PDFs are saved here. Re-ingesting the same file is skipped unless you force it. |
| 2 | **PDF loader** | [rag/pdf_loader.py](rag/pdf_loader.py) | Reads the embedded text layer, pulls out tables and figure captions, and flags pages that need OCR. |
| 3 | **OCR (Surya)** | [rag/ocr.py](rag/ocr.py) | Re-reads scanned or RTL (Arabic/Hebrew) pages as images. Results are cached under `data/cache/ocr`. |
| 4 | **Chunker** | [rag/chunker.py](rag/chunker.py) | Splits pages into structure-aware chunks with overlap. Optional parent–child windows give broader context at answer time. |
| 5 | **QA extractor** | [rag/qa_extractor.py](rag/qa_extractor.py) | An LLM mines questions, answers, evidence quotes, paraphrases, and difficulty labels from each chunk. Pairs without verifiable evidence are dropped. |
| 6 | **Embed** | [rag/ollama_client.py](rag/ollama_client.py) | `bge-m3` produces 1024-dim embeddings for both questions and chunks. |
| 7 | **Vector store** | [rag/vectorstore.py](rag/vectorstore.py) | Persists two Chroma collections (question bank + document chunks) and in-memory BM25 indexes for hybrid search. |

**Guardrails (manual Q/A):** From the Streamlit UI you can add or edit question–answer pairs at any time without re-ingesting. Thumbs-down on a live answer parks it in the question bank as **Needs review** so you can fix it and approve it into the bank; thumbs-up saves nothing. Approved entries are indexed like extracted ones and take priority in Tier 1 retrieval — they steer the system toward approved phrasing and grounding.

---

### Query path (answer many times)

Routes each user question through the bank first, then chunk search, then refusal.

| # | Part | Module | What it does |
|---|---|---|---|
| 1 | **User query** | — | Natural-language question from the UI, API, or CLI. |
| 2 | **Retrieval router** | [rag/retrieval.py](rag/retrieval.py) | Decomposes complex queries, runs hybrid dense + BM25 search, reranks candidates with an LLM, and applies a relevance gate. |
| 3 | **Tier 1 — Question bank** | [rag/vectorstore.py](rag/vectorstore.py) | Matches against extracted and manual questions (and paraphrases). Hit threshold: **≥ 0.72**. Returns pre-grounded answers with evidence. |
| 4 | **Tier 2 — Classical RAG** | [rag/vectorstore.py](rag/vectorstore.py) | Falls back to reciprocal-rank fusion over document chunks when Tier 1 misses. Hit threshold: **≥ 0.45**. |
| 5 | **Tier 3 — Refuse** | [rag/retrieval.py](rag/retrieval.py) | If neither tier clears its threshold, the system refuses without calling the LLM — no answer is better than a hallucinated one. |
| 6 | **Answerer** | [rag/answerer.py](rag/answerer.py) | Synthesises a final response with page-level citations, uncertainty badges, and a post-hoc groundedness audit on Tier 2 answers. |

---

### Local inference (Ollama)

| Part | Module | What it does |
|---|---|---|
| **Ollama** | [rag/ollama_client.py](rag/ollama_client.py) | All LLM and embedding calls stay on your machine. Default models: `gemma4:31b` (extract + answer) and `bge-m3` (embeddings). Override with `RAG_EXTRACT_MODEL` / `RAG_ANSWER_MODEL`. |

---

### Hallucination guards and observability

| Guard / tool | Where | What it does |
|---|---|---|
| **Evidence at extraction** | [rag/qa_extractor.py](rag/qa_extractor.py) | Every Q/A pair must quote its source verbatim; unverifiable pairs never enter the index. |
| **Similarity thresholds** | [rag/retrieval.py](rag/retrieval.py) | Below threshold → no retrieval, no generation. |
| **`INSUFFICIENT_CONTEXT` sentinel** | [rag/answerer.py](rag/answerer.py) | The answering prompt forbids outside knowledge; the model must refuse when sources fall short. |
| **Groundedness audit** | [rag/answerer.py](rag/answerer.py) | A second LLM pass checks each claim against retrieved context before the answer is shown. |
| **Manual Q/A guardrails** | [app.py](app.py) | Curated bank entries steer Tier 1 toward trusted answers. |
| **OCR + extract caches** | `data/cache/` | Speed up re-ingest when document text has not changed. |
| **Eval harness** | [eval/](eval/) | Route-only and baseline comparisons against naive vector RAG. |
| **Metrics** | [rag/metrics.py](rag/metrics.py) | Tracks refuse rate and latency over time. |

### Shared configuration

| File | Role |
|---|---|
| [config.py](config.py) | Every tunable setting, overridable via environment variables |
| [rag/schemas.py](rag/schemas.py) | Dataclass contracts (`Chunk`, `QAPair`, `AnswerResult`, etc.) shared by all stages |
| [rag/text_normalize.py](rag/text_normalize.py) | Arabic-aware tokenisation for BM25 and deduplication |

Citations resolve to a document name, a page label, and the verbatim quote the
claim rests on — every answer is auditable back to the page it came from.
