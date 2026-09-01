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

| File | Role |
|---|---|
| [config.py](config.py) | Every tunable, env-overridable |
| [rag/schemas.py](rag/schemas.py) | Dataclass contracts shared by all stages |
| [rag/ollama_client.py](rag/ollama_client.py) | Chat, schema-constrained JSON, normalised embeddings |
| [rag/pdf_loader.py](rag/pdf_loader.py) | PDF → clean page-attributed text |
| [rag/chunker.py](rag/chunker.py) | Structure-aware, page-tracking chunking |
| [rag/qa_extractor.py](rag/qa_extractor.py) | The question bank miner + evidence validation |
| [rag/vectorstore.py](rag/vectorstore.py) | Two Chroma collections, hybrid dense+BM25 |
| [rag/retrieval.py](rag/retrieval.py) | The tiered router |
| [rag/answerer.py](rag/answerer.py) | Grounded synthesis, citations, groundedness check |
| [rag/pipeline.py](rag/pipeline.py) | Façade + CLI |
| [app.py](app.py) | Streamlit UI |

Citations resolve to a document name, a page label, and the verbatim quote the
claim rests on — every answer is auditable back to the page it came from.
