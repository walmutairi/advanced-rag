"""Central configuration for the Advanced RAG system.

Every tunable lives here so the pipeline can be reshaped without touching
module code. Values can be overridden with environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = DATA_DIR / "cache"

for _d in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

#: Legacy alias — sets both extract and answer models when the specific vars
#: below are not provided.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gemma4:31b")

#: Ingest-time question extraction. Defaults to ``LLM_MODEL``; point at a
#: smaller model (e.g. ``gemma3:4b``) for much faster indexing.
EXTRACT_MODEL = os.getenv("RAG_EXTRACT_MODEL", LLM_MODEL)

#: Query-time answering, reranking and groundedness checks.
ANSWER_MODEL = os.getenv("RAG_ANSWER_MODEL", LLM_MODEL)

#: Embedding model. bge-m3 produces 1024-dim multilingual embeddings and
#: markedly outperforms nomic-embed-text on retrieval benchmarks.
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "bge-m3:latest")

#: Generation is deliberately near-greedy: this is an extraction/grounding
#: task, not a creative one. gemma4 ships with temperature=1 by default.
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.15"))
LLM_NUM_CTX = int(os.getenv("RAG_LLM_NUM_CTX", "16384"))

#: Long-running local generation on a 31B model needs a generous ceiling.
OLLAMA_TIMEOUT = int(os.getenv("RAG_OLLAMA_TIMEOUT", "1800"))
OLLAMA_MAX_RETRIES = int(os.getenv("RAG_OLLAMA_MAX_RETRIES", "3"))

#: Per-chunk / cross-chunk extract calls. Keep this much lower than
#: ``OLLAMA_TIMEOUT`` so one hung Ollama request cannot freeze ingest for 30–90
#: minutes (timeout × retries × chat_json attempts).
EXTRACT_TIMEOUT = int(os.getenv("RAG_EXTRACT_TIMEOUT", "300"))

# --------------------------------------------------------------------------
# OCR (Surya)
# --------------------------------------------------------------------------
# PyMuPDF's embedded-text extraction is fast and exact for well-formed Latin
# PDFs, but it returns RTL spans in visual order and mis-decomposes Arabic
# lam-alef ligatures. Surya re-reads the page image and emits correct logical
# order, so it is the authority whenever the text layer is absent or suspect.

ENABLE_OCR = os.getenv("RAG_ENABLE_OCR", "1") == "1"

#: Render resolution for pages handed to OCR. 200 balances accuracy against
#: memory; below ~150 recognition of small type degrades noticeably.
OCR_DPI = int(os.getenv("RAG_OCR_DPI", "200"))

#: "auto"  — OCR only pages whose text layer is missing or judged unreliable
#: "never" — trust the embedded text layer unconditionally
#: "always"— OCR every page regardless of the text layer
OCR_MODE = os.getenv("RAG_OCR_MODE", "auto")

#: A page with fewer than this many extractable characters is treated as
#: having no usable text layer. This is a PER-PAGE test; a short document is
#: not a scanned one, so document-level character counts must never gate OCR.
OCR_MIN_CHARS_PER_PAGE = int(os.getenv("RAG_OCR_MIN_CHARS_PER_PAGE", "80"))

#: Fraction of a page's characters that must be RTL script before the text
#: layer is considered unreliable. PyMuPDF's RTL span ordering is broken, so
#: any substantially Arabic/Hebrew page is better served by OCR.
OCR_RTL_TRIGGER_RATIO = float(os.getenv("RAG_OCR_RTL_TRIGGER_RATIO", "0.15"))

#: Drop OCR lines below this confidence — they are usually noise from
#: figures, stamps or page furniture.
OCR_MIN_CONFIDENCE = float(os.getenv("RAG_OCR_MIN_CONFIDENCE", "0.50"))

#: torch device for Surya. "auto" picks mps on Apple Silicon, else cpu.
OCR_DEVICE = os.getenv("RAG_OCR_DEVICE", "auto")

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

CHUNK_TARGET_CHARS = int(os.getenv("RAG_CHUNK_TARGET_CHARS", "3600"))
CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "400"))
CHUNK_MIN_CHARS = int(os.getenv("RAG_CHUNK_MIN_CHARS", "250"))

#: Build parent section windows over child chunks (retrieve child, answer with parent).
ENABLE_PARENT_CHILD_CHUNKS = os.getenv("RAG_ENABLE_PARENT_CHILD_CHUNKS", "1") == "1"
PARENT_CHILD_WINDOW = max(2, int(os.getenv("RAG_PARENT_CHILD_WINDOW", "3")))

#: Extra multi_hop pass over adjacent chunk pairs after per-chunk extraction.
#: Off by default: neighbor context already covers most multi_hop cases, and
#: this pass can stall a 31B extract for tens of minutes with no bar movement.
ENABLE_CROSS_CHUNK_SYNTHESIS = os.getenv("RAG_ENABLE_CROSS_CHUNK_SYNTHESIS", "0") == "1"

#: Cap adjacent pairs synthesised when the pass is enabled (avoid O(n) LLM calls).
CROSS_CHUNK_MAX_PAIRS = max(1, int(os.getenv("RAG_CROSS_CHUNK_MAX_PAIRS", "12")))

# --------------------------------------------------------------------------
# Question extraction
# --------------------------------------------------------------------------

#: How many Q/A pairs to request per chunk. The extractor is told to emit
#: fewer if the chunk genuinely does not support this many.
QA_PER_CHUNK_MIN = int(os.getenv("RAG_QA_PER_CHUNK_MIN", "3"))
QA_PER_CHUNK_MAX = int(os.getenv("RAG_QA_PER_CHUNK_MAX", "7"))

#: Paraphrases indexed alongside each canonical question. These widen the
#: surface a user query can hit without diluting the answer itself.
QA_PARAPHRASES = int(os.getenv("RAG_QA_PARAPHRASES", "3"))

#: Chunks shorter than this are skipped without an LLM call. Keep it LOW:
#: character counts are a poor proxy for information content across scripts.
#: 97 characters of Arabic carried four extractable facts in testing, where
#: the same count of English prose would be a fragment. Too high a value
#: silently empties the question bank for short or non-Latin documents.
QA_MIN_CHUNK_CHARS = int(os.getenv("RAG_QA_MIN_CHUNK_CHARS", "60"))

#: Parallel LLM extraction workers during ingest. Keep at 1 with a single
#: Ollama runner slot (``-np 1``): concurrent chats often queue forever and
#: look like a stuck 95% bar.
QA_EXTRACT_WORKERS = max(1, int(os.getenv("RAG_QA_EXTRACT_WORKERS", "1")))

#: Include previous/next chunk text in the extraction prompt so multi_hop
#: questions can join adjacent passages without a second corpus pass.
EXTRACT_NEIGHBOR_CHUNKS = os.getenv("RAG_EXTRACT_NEIGHBOR_CHUNKS", "1") == "1"

#: Reuse extraction results when the chunk text hash has not changed.
ENABLE_EXTRACT_CACHE = os.getenv("RAG_ENABLE_EXTRACT_CACHE", "1") == "1"

#: Question-bank browser page size in the Streamlit UI.
QUESTION_BANK_PAGE_SIZE = max(10, int(os.getenv("RAG_QUESTION_BANK_PAGE_SIZE", "50")))

#: Ollama /api/embed request size. Larger batches cut round-trips on ingest.
EMBED_BATCH_SIZE = max(1, int(os.getenv("RAG_EMBED_BATCH_SIZE", "32")))

#: Persist OCR page text under data/cache/ocr keyed by content hash + page + DPI.
ENABLE_OCR_CACHE = os.getenv("RAG_ENABLE_OCR_CACHE", "1") == "1"

#: Cognitive levels the extractor must cover. Driving variety here is what
#: makes the question bank "advanced" rather than a list of definitions.
QUESTION_TYPES = [
    "factual",       # directly stated in the text
    "definitional",  # what is X / terminology
    "conceptual",    # why / how does it work
    "causal",        # cause-effect relationships
    "comparative",   # X versus Y, trade-offs
    "quantitative",  # numbers, thresholds, measurements
    "procedural",    # steps, sequences, methods
    "multi_hop",     # requires joining two or more separate statements
    "application",   # apply the material to a novel scenario
    "critical",      # limitations, assumptions, implications
]

DIFFICULTY_LEVELS = ["basic", "intermediate", "advanced"]

# --------------------------------------------------------------------------
# Retrieval routing thresholds
# --------------------------------------------------------------------------
# Scores are cosine similarity in [0, 1]; higher is more similar.

#: Tier 1 accept: a question-bank hit at or above this is treated as a
#: direct match to a pre-extracted question.
QUESTION_MATCH_THRESHOLD = float(os.getenv("RAG_QUESTION_MATCH_THRESHOLD", "0.72"))

#: Every additional question-bank hit within this margin of the best hit is
#: also returned, so multi-answer questions surface all of their answers.
QUESTION_MULTI_MARGIN = float(os.getenv("RAG_QUESTION_MULTI_MARGIN", "0.08"))

#: Tier 2 accept: minimum chunk similarity for classical RAG to even try.
#: Below this we refuse rather than let the model improvise.
VECTOR_MATCH_THRESHOLD = float(os.getenv("RAG_VECTOR_MATCH_THRESHOLD", "0.45"))

QUESTION_TOP_K = int(os.getenv("RAG_QUESTION_TOP_K", "12"))
VECTOR_TOP_K = int(os.getenv("RAG_VECTOR_TOP_K", "8"))

#: Weight of dense (embedding) score against sparse BM25 score in the
#: hybrid retriever. 1.0 = pure dense, 0.0 = pure BM25.
HYBRID_DENSE_WEIGHT = float(os.getenv("RAG_HYBRID_DENSE_WEIGHT", "0.65"))

#: Ask the LLM to rerank retrieved candidates before answering.
ENABLE_LLM_RERANK = os.getenv("RAG_ENABLE_LLM_RERANK", "1") == "1"

#: Verify the drafted answer is entailed by the retrieved context before
#: showing it. Costs one extra LLM call; it is the main hallucination guard.
ENABLE_GROUNDEDNESS_CHECK = os.getenv("RAG_ENABLE_GROUNDEDNESS_CHECK", "1") == "1"

#: Fraction of the drafted answer's atomic claims that must be supported by the
#: retrieved sources. Defaults to 1.0 — every claim — because the requirement
#: this system is built around is absolute: no assertion the documents do not
#: make. At the previous 0.5 an answer with half its claims fabricated was
#: published, which is worse than a refusal because the true half lends the
#: false half credibility. Lower it only if you would rather over-answer than
#: over-refuse, and know that is the trade you are making.
GROUNDEDNESS_MIN_SCORE = float(os.getenv("RAG_GROUNDEDNESS_MIN_SCORE", "1.0"))

#: Skip the post-hoc groundedness LLM call for strong question-bank hits —
#: evidence quotes were already validated at extraction time.
SKIP_GROUNDEDNESS_ON_QUESTION_BANK = (
    os.getenv("RAG_SKIP_GROUNDEDNESS_ON_QUESTION_BANK", "1") == "1"
)

#: Minimum confidence at which Tier-1 answers skip groundedness.
QUESTION_BANK_SKIP_GROUNDEDNESS_MIN = float(
    os.getenv("RAG_QUESTION_BANK_SKIP_GROUNDEDNESS_MIN", "0.85")
)

#: When Tier 1 accepts but confidence is below this ceiling, also retrieve
#: Tier 2 chunks and answer from both (hybrid route).
ENABLE_HYBRID_ROUTE = os.getenv("RAG_ENABLE_HYBRID_ROUTE", "1") == "1"
HYBRID_TIER1_CEILING = float(os.getenv("RAG_HYBRID_TIER1_CEILING", "0.85"))

#: LLM chunk rerank after hybrid dense+BM25 (uses ANSWER_MODEL / small model).
ENABLE_CHUNK_RERANK = os.getenv("RAG_ENABLE_CHUNK_RERANK", "1") == "1"

#: Split multi-part questions into sub-queries for Tier 2.
ENABLE_QUERY_DECOMPOSITION = os.getenv("RAG_ENABLE_QUERY_DECOMPOSITION", "1") == "1"

#: Soften thresholds slightly from observed score distribution (never below floor).
ENABLE_ADAPTIVE_THRESHOLDS = os.getenv("RAG_ENABLE_ADAPTIVE_THRESHOLDS", "1") == "1"
ADAPTIVE_THRESHOLD_FLOOR_RATIO = float(
    os.getenv("RAG_ADAPTIVE_THRESHOLD_FLOOR_RATIO", "0.92")
)

#: Run golden-question eval automatically after each successful ingest.
ENABLE_POST_INGEST_EVAL = os.getenv("RAG_ENABLE_POST_INGEST_EVAL", "1") == "1"
EVAL_QUESTIONS_PATH = os.getenv(
    "RAG_EVAL_QUESTIONS_PATH",
    str(ROOT_DIR / "eval" / "sample_questions.json"),
)

#: Require every answer sentence to carry a citation marker when possible.
ENABLE_CLAIM_CITATIONS = os.getenv("RAG_ENABLE_CLAIM_CITATIONS", "1") == "1"

# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------

QUESTION_COLLECTION = "extracted_questions"
CHUNK_COLLECTION = "document_chunks"

#: Exact wording used whenever the system declines to answer. Kept in one
#: place so the UI and the answerer can never drift apart.
REFUSAL_MESSAGE = (
    "I could not find the answer to your question in the provided documents."
)
