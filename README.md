# 📚 Multimodal RAG Pipeline over AI Research Papers

A production-grade Retrieval-Augmented Generation (RAG) system built over foundational AI research papers covering LLMs, RAG, Transformers, Embeddings, and AI Agents. The pipeline handles both text and images, combining hybrid search, neural reranking, and multimodal generation to deliver accurate, cited answers.

---

## 🖥️ Demo

> Ask questions like:
> - *"What is the BM25 weighting scheme?"*
> - *"How does the attention mechanism work in transformers?"*
> - *"What are the components of an LLM-based agent?"*
> - *"How does RAG differ from fine-tuning?"*


<img width="694" height="700" alt="Screenshot 2026-07-16 at 19 38 53" src="https://github.com/user-attachments/assets/9886ce03-fe21-4733-bba2-292df37621b9" />
<img width="694" height="700" alt="Screenshot 2026-07-16 at 19 30 28" src="https://github.com/user-attachments/assets/a3815545-7f98-4aed-9481-671c2209cbbf" />


---

## 🏗️ Architecture

```
                        OFFLINE (ingest.py)
----------------------------------------------------------------------------
│  PDF → Unstructured → Chunks       +             Images                  │
│                        │                          │                      │
│                   Text Chunks                 Image Elements             │
│                        │                          │                      │
│                  BGE Embeddings           Gemini Vision → Summary        │
│                 miniCOIL Sparse                   │                      │
│                       │                   BGE Embeddings                 │
│                       ----------------------------┘                      │
│                            Qdrant (Docker)                               │
└--------------------------------------------------------------------------┘

                        ONLINE (pipeline.py)
┌───────────────────────────────────────────────────────── ┐
│  User Query                                              │
│      │                                                   │
│  Hybrid Search (dense + sparse, k=15)                    │
│      │                                                   │
│  Score Threshold Filter (>0.2)                           │
│      │                                                   │
│  Cohere Reranker (text chunks only, k=15→3)              │
│      │                    │                              │
│  Top 3 Text Chunks   Image Chunks (pass-through)         │
│      └──────────────────--┘                              │
│                    │                                     │
│             Gemini Flash                                 │
│         (text + actual images)                           │
│                    │                                     │
│         Answer with Citations                            │
└─────────────────────────────────────────────────────────-┘
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


## ⚡ Performance

Measured on Apple M2 8GB, Qdrant running locally via Docker (last 10 queries):

| Stage | Avg | Min | Max |
|---|---|---|---|
| Hybrid Retrieval (k=15, score threshold >0.2) | 0.40s | 0.11s | 1.07s |
| Cohere Reranking (k=15→3) | 0.49s | 0.37s | 0.88s |
| Gemini Flash Generation | 1.63s | 1.41s | 2.03s |
| **Total** | **~2.5s** | **~2s** | **~4s** |

Retrieval variance (0.11s→1.07s) is due to Qdrant cold start on first query — subsequent queries are consistently fast. Generation is stable (1.41-2.03s) reflecting Gemini Flash's consistent response times.

Latency traced via LangSmith `@traceable` decorators — actual measured wall clock time per pipeline step.

----

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
## 📁 Project Structure

```
RAG_PROJECT/
├── Backend/
│   ├── extract_chunk.py    # PDF parsing, chunking
│   ├── ingest.py           # embed + store in Qdrant
│   └── pipeline.py         # retrieval, reranking, generation
├── Eval_Script/
│   └── evaluate.py         # DeepEval evaluation
├── Frontend/
│   └── app.py              # Streamlit
├── json_files/
│   ├── chunks.json         # processed chunks
│   ├── eval_dataset.json   # 25 evaluation questions
│   └── deepeval_results.json
├── Papers/                 # source PDFs
├── main.py                 # FastAPI
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF Parsing and Chunking| Unstructured | Best-in-class PDF element extraction including images, provides chunking by title |
| Dense Embeddings | BAAI/bge-base-en-v1.5 | Optimized for asymmetric retrieval (query→passage), 512 token limit |
| Sparse Embeddings | Qdrant/minicoil-v1 | Learned sparse model with native Qdrant integration, no RAM overhead unlike BM25Retriever. Enables single-call hybrid search with built-in RRF fusion. Also better at handling morphological variants of terms |
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
results = [r for r in results if r[1] > threshold] threshold=0.2
```

Filtering at retrieval reduces noise passed to the reranker and improves reranking quality by ensuring only genuinely candidate chunks are scored.

### Separation of Concerns
The pipeline is split into two distinct processes:

- **`ingest.py`** (offline) — document processing, chunking, embedding, indexing. Run once when adding new documents.
- **`pipeline.py`** (online) — query processing, retrieval, reranking, generation. All models initialized at module level, loaded once on server startup.

This is standard production RAG architecture — ingestion and serving never share a process.

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

<img width="1219" height="793" alt="Screenshot 2026-07-16 at 18 31 49" src="https://github.com/user-attachments/assets/b792cf84-b323-4977-8547-0e92084e7c25" />
<img width="1219" height="793" alt="Screenshot 2026-07-16 at 18 30 26" src="https://github.com/user-attachments/assets/80bcbcbb-ba35-415f-a020-d4b0374344f7" />

---

## 📖 Corpus Statistics

| Stat | Value |
|---|---|
| Papers indexed | 6 |
| Total pages | 149 |
| Total chunks | 471 |
| Image chunks | 81 |
| Text chunks | 390 |
| Chunk size | 1000 chars |
| Chunk overlap | 150 chars |
| Embedding dimensions | 768 (BGE-base) |


---

## 📈 Retrieval Benchmarks

**Hybrid vs Dense only (observed during development):**

| Approach | Rank of correct chunk (BM25 query) |
|---|---|
| Dense only | Rank 1-2 ✅ |
| Hybrid k=15 (no reranker) | Rank 7-8 ❌ (RRF demotion) |
| Hybrid k=15 + Cohere reranker (→3) | Rank 1-3 ✅ |

RRF demotion was a real observed failure — hybrid search penalized semantically relevant chunks that sparse search didn't match, dragging their RRF score down. Cohere reranker resolved this by scoring query-chunk pairs directly rather than relying on rank fusion math.

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
- **CI/CD Evaluation** — automated DeepEval evaluation on every push via GitHub Actions, blocking merges if metrics drop below threshold (faithfulness < 0.85, contextual precision < 0.85). Requires Qdrant Cloud migration for GitHub Actions accessibility.

---
