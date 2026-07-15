# 📚 Multimodal RAG Pipeline over AI Research Papers

A production-grade Retrieval-Augmented Generation (RAG) system built over foundational AI research papers covering LLMs, RAG, Transformers, Embeddings, and AI Agents. The pipeline handles both text and images, combining hybrid search, neural reranking, and multimodal generation to deliver accurate, cited answers.

---

## 🖥️ Demo

> Ask questions like:
> - *"What is the BM25 weighting scheme?"*
> - *"How does the attention mechanism work in transformers?"*
> - *"What are the components of an LLM-based agent?"*
> - *"How does RAG differ from fine-tuning?"*

---

## 🏗️ Architecture

```
                        OFFLINE (ingest.py)
┌─────────────────────────────────────────────────────────┐
│  PDF → Unstructured → Chunks + Images                   │
│           │                    │                         │
│     Text Chunks          Image Elements                  │
│           │                    │                         │
│     BGE Embeddings    Gemini Vision → Summary            │
│     miniCOIL Sparse          │                           │
│           │              BGE Embeddings                  │
│           └──────────────────┘                           │
│                    Qdrant (Docker)                        │
└─────────────────────────────────────────────────────────┘

                        ONLINE (pipeline.py)
┌─────────────────────────────────────────────────────────┐
│  User Query                                             │
│      │                                                   │
│  Hybrid Search (dense + sparse, k=20)                   │
│      │                                                   │
│  Score Threshold Filter                                  │
│      │                                                   │
│  Cohere Reranker (text chunks only)                      │
│      │                    │                              │
│  Top 5 Text Chunks   Image Chunks (pass-through)         │
│      └──────────────────┘                                │
│                    │                                     │
│             Gemini Flash                                 │
│         (text + actual images)                           │
│                    │                                     │
│         Answer with Citations                            │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Multimodal ingestion** — Gemini Vision processes images during indexing, generating context-aware summaries
- **Hybrid search** — combines dense semantic search (BGE) and sparse keyword search (miniCOIL) via Qdrant's native RRF fusion
- **Neural reranking** — Cohere rerank-english-v3.0 rescores retrieved chunks for precise relevance ordering
- **Multimodal generation** — actual images passed to Gemini alongside text for visually-grounded answers
- **Source citations** — every answer references the source file and page number
- **Observability** — full pipeline tracing via LangSmith
- **Evaluation** — DeepEval evaluation framework with 25-question golden dataset

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF Parsing | Unstructured | Best-in-class PDF element extraction including images |
| Dense Embeddings | BAAI/bge-base-en-v1.5 | Optimized for asymmetric retrieval (query→passage), 512 token limit |
| Sparse Embeddings | Qdrant/minicoil-v1 | Learned sparse model, better than BM25 for technical terms |
| Vector Store | Qdrant (Docker) | Native hybrid search + RRF fusion, no RAM overhead |
| Reranker | Cohere rerank-english-v3.0 | Best-in-class reranking quality, domain-generalizable |
| Generation | Gemini Flash | Multimodal, fast, cost-effective |
| Observability | LangSmith | Full pipeline tracing per query |
| Evaluation | DeepEval | LLM-as-judge evaluation metrics |
| Backend | FastAPI | Production-grade API with Pydantic validation |
| Frontend | Streamlit | Clean UI without frontend complexity |

---

## 🔬 Pipeline Deep Dive

### Chunking Strategy
Chunks are set to **1000 characters with 150 character overlap** — a deliberate choice informed by embedding model constraints:

- `all-MiniLM-L6-v2` (initial model) has a **256 token limit (~1024 chars)**. Our original 2000-char chunks were being **silently truncated**, meaning half of every chunk was never embedded.
- Beyond truncation, large chunks cause **embedding dilution** — when a chunk covers multiple topics, its vector sits at the "average" of all topics rather than cleanly representing any one of them. A query about LLMs would fail to retrieve a chunk that was half about LLMs and half about attention mechanisms.
- Switching to 1000 chars + BGE-base (512 token limit) resolved rank-1 retrieval for definitional queries.

### Image Handling Strategy
Images are handled via a **dual-chunk architecture**:

```
Image extracted → sent to Gemini Vision with surrounding text
                → generates context-aware summary
                → stored as separate chunk linked to parent via source_id

Retrieval:  image summary chunk embedded → searchable via text queries
Generation: actual image (base64) passed to Gemini → visually-grounded answer
```

This makes images **findable via semantic search** (through their text summary) while ensuring **accurate reasoning** at generation time (through the actual image).

### Hybrid Search
Qdrant's native hybrid search runs **both dense and sparse retrieval in parallel**, combining results via **Reciprocal Rank Fusion (RRF)**:

```
RRF score = 1 / (rank + 60)
```

RRF rewards chunks that rank highly in **both** searches without requiring manual weight tuning. A chunk ranking 1st in dense and 1st in sparse scores significantly higher than one ranking 1st in only one.

Dense search handles semantic meaning ("what is a language model"), sparse search handles exact technical terms ("RLHF", "LoRA", "POMDP") that semantic search can miss.

### Reranking Design
A cross-encoder reranker was added to address **RRF demotion** — cases where a semantically relevant chunk was ranked lower because sparse search didn't match it well, dragging down its RRF score.

**Why Cohere over local cross-encoders:**
- `ms-marco-MiniLM-L-6-v2` was evaluated first but failed on academic text — it was trained on MS MARCO (web search queries + short passages), causing domain mismatch with dense academic prose
- Cohere's `rerank-english-v3.0` is trained on diverse data with better domain generalization, confirmed by improved chunk rankings on academic content

**Image chunks are excluded from reranking** — cross-encoders are fooled by IR vocabulary in image summaries (words like "precision", "recall", "rank" appear in summaries describing IR evaluation figures, causing false matches to IR-related queries). Image chunks bypass reranking and pass directly to generation if hybrid search surfaced them, since RRF already judged them relevant.

### Score Threshold Filtering
A similarity score threshold is applied **at retrieval time** (not downstream) — single place to tune, cleaner pipeline:

```python
results = [r for r in results if r[1] > threshold]
```

Filtering at retrieval reduces noise passed to the reranker and improves reranking quality by ensuring only genuinely candidate chunks are scored.

### Separation of Concerns
The pipeline is split into two distinct processes:

- **`ingest.py`** (offline) — document processing, chunking, embedding, indexing. Run once when adding new documents.
- **`pipeline.py`** (online) — query processing, retrieval, reranking, generation. All models initialized at module level, loaded once on server startup.

This is standard production RAG architecture — ingestion and serving never share a process.

---

## 📊 Evaluation Results

Evaluated using **DeepEval** with a 25-question golden dataset generated from the corpus. Questions were generated using Gemini with Pydantic structured output and a **context quality gate** — each chunk was assessed by the LLM for self-contained context before question generation, filtering out mid-sentence continuation chunks.

| Metric | Score | Pass Rate |
|---|---|---|
| Faithfulness | 0.95 | 92% |
| Answer Relevancy | 0.85 | 84% |
| Contextual Precision | 0.92 | 96% |
| Contextual Recall | 0.94 | 88% |

**Note on Answer Relevancy (0.85):** Scores are slightly lower due to the evaluation metric penalizing source citations (page numbers, file references) included in responses for traceability. Citations are an intentional feature of the pipeline — the evaluator limitation was identified and documented rather than removed to chase a higher score.

---

## ⚠️ Known Limitations

**Image fragment rendering** — Unstructured sometimes splits composite figures spanning multiple PDF elements into separate image fragments. Full image reconstruction would require custom PDF rendering logic (e.g. page-level rasterization + figure boundary detection).

**Formula extraction** — Mathematical formulas in PDFs are extracted as garbled unicode characters by Unstructured. Formula-heavy chunks have degraded embedding quality and retrieval accuracy. A dedicated math OCR pipeline (e.g. Mathpix) would be needed for accurate formula handling.

**Text-based image retrieval** — Image retrieval is text-based (over LLM-generated summaries), not visual. True multimodal retrieval would require visual embeddings (e.g. CLIP) to search over image content directly. Current approach: Level 1 (vision-aware ingestion) + Level 3 (multimodal generation), missing Level 2 (visual retrieval).

**Cross-encoder domain mismatch** — Standard cross-encoders trained on MS MARCO (web search) underperform on academic text. Cohere's hosted reranker mitigates this but introduces an external API dependency.

---

## 🚀 Future Improvements

- **Visual embeddings** — CLIP-based image embeddings for true multimodal retrieval, not just text-based image search
- **GraphRAG** — Knowledge graph construction for multi-hop reasoning across papers (e.g. "how does attention in transformers relate to retrieval in RAG?")
- **Job queue** — RQ/Celery + Redis for concurrent user handling at scale. Currently FastAPI handles requests synchronously, suitable for low traffic.
- **HyDE** — Hypothetical Document Embedding for asymmetric queries where dense retrieval still struggles. Validated manually, skipped as retrieval quality was already sufficient.
- **Math OCR** — Mathpix or similar for accurate formula extraction and embedding
- **Multimodal reranker** — Cross-encoders that understand visual content for proper image chunk reranking

---

## 📁 Project Structure

```
project/
├── ingest.py          # Offline: PDF parsing, chunking, embedding, indexing
├── pipeline.py        # Online: retrieval, reranking, generation
├── main.py            # FastAPI backend
├── frontend/
│   └── app.py         # Streamlit frontend
├── evaluation/
│   ├── generate_dataset.py   # QA dataset generation
│   ├── evaluate.py           # DeepEval evaluation
│   ├── ragas_dataset.json    # Generated test questions
│   └── deepeval_results.csv  # Evaluation scores
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Docker (for Qdrant)
- Python 3.10+
- API keys: Gemini, Cohere, LangSmith

### 1. Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
GOOGLE_API_KEY=your_key
COHERE_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-pipeline
```

### 4. Run ingestion (once)
```bash
python ingest.py
```

### 5. Start backend
```bash
uvicorn main:app --reload
```

### 6. Start frontend
```bash
streamlit run frontend/app.py
```

---

## 🔭 Observability

All pipeline steps are traced via **LangSmith** with `@traceable` decorators:

```
RAG Pipeline
├── Chunks Retrieved    → query, k, chunks returned, latency
├── Chunks Reranked     → query, input chunks, reranked output, latency
└── Answer Generated    → prompt, chunks used, answer, tokens, latency
```

LangSmith enables per-query debugging — when evaluation scores are low, traces show exactly which retrieval or generation step caused the failure.

---

## 📖 Papers Indexed

- Foundational LLMs
- Retrieval-Augmented Generation (RAG)
- Transformer Architecture
- Embeddings and Vector Representations
- AI Agents
- Vector Databases
