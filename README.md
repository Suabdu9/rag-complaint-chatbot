# 🏦 CrediTrust Complaint Intelligence Chatbot

> A RAG-powered internal tool that transforms raw CFPB customer complaint narratives into strategic, evidence-backed insights for Product, Support, and Compliance teams.

---

## Overview

CrediTrust Financial receives thousands of customer complaints every month. This project builds a **Retrieval-Augmented Generation (RAG)** pipeline that lets non-technical stakeholders ask plain-English questions and receive synthesised, grounded answers — in seconds instead of days.

**Covered product categories:**
- Credit Card
- Personal Loan
- Savings Account
- Money Transfer

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────┐
│           Embedding Model               │
│     (all-MiniLM-L6-v2, 384-dim)        │
└────────────────┬────────────────────────┘
                 │ query vector
                 ▼
┌─────────────────────────────────────────┐
│         Vector Store                    │
│     ChromaDB / FAISS                    │
│   ~1.37M complaint chunks               │
└────────────────┬────────────────────────┘
                 │ top-k relevant chunks
                 ▼
┌─────────────────────────────────────────┐
│         Prompt Builder                  │
│  Injects chunks + question into         │
│  structured analyst prompt template     │
└────────────────┬────────────────────────┘
                 │ grounded prompt
                 ▼
┌─────────────────────────────────────────┐
│         LLM Generator                   │
│    (Mistral-7B-Instruct or similar)     │
└────────────────┬────────────────────────┘
                 │ answer + sources
                 ▼
        Gradio Chat Interface
```

---

## Project Structure

```
rag-complaint-chatbot/
├── .github/workflows/unittests.yml   # CI pipeline
├── data/
│   ├── raw/                          # Raw CFPB CSV (not committed)
│   └── processed/                    # filtered_complaints.csv, sample, figures
├── vector_store/                     # ChromaDB + FAISS index (not committed)
├── notebooks/
│   ├── task1_eda_preprocessing.py    # EDA + cleaning pipeline
│   ├── task2_chunking_embedding.py   # Chunking + embedding + indexing
│   └── task3_rag_evaluation.py       # Evaluation harness
├── src/
│   └── rag_pipeline.py               # Core RAG logic
├── tests/
│   └── test_pipeline.py              # Unit tests
├── app.py                            # Gradio UI
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/your-org/rag-complaint-chatbot.git
cd rag-complaint-chatbot

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your HuggingFace token and paths
```

### 2. Add data

Place the CFPB complaint CSV at `data/raw/complaints.csv`.
Or place the pre-built `complaint_embeddings.parquet` at `data/raw/`.

### 3. Run Task 1 — EDA & Preprocessing

```bash
python notebooks/task1_eda_preprocessing.py
# Outputs: data/processed/filtered_complaints.csv
#          data/processed/figures/*.png
```

### 4. Run Task 2 — Chunking & Embedding

```bash
python notebooks/task2_chunking_embedding.py
# Outputs: vector_store/chroma/  (ChromaDB)
#          vector_store/faiss_index.bin  (FAISS)
```

> **Shortcut:** Use the pre-built `complaint_embeddings.parquet` from the challenge resources and point `CHROMA_PERSIST_DIR` to the pre-built ChromaDB directory.

### 5. Run Task 3 — Evaluation

```bash
python notebooks/task3_rag_evaluation.py
```

### 6. Launch the Chatbot UI

```bash
python app.py
# Opens at http://localhost:7860
```

### 7. Run Tests

```bash
pytest tests/ -v --cov=src
```

---

## Configuration

All key settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL_ID` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `LLM_MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | Text generation LLM |
| `VECTOR_STORE_TYPE` | `chroma` | `chroma` or `faiss` |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Character overlap between chunks |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | `all-MiniLM-L6-v2` | Fast, 80 MB, 384-dim, excellent semantic similarity |
| Vector DB | ChromaDB (primary) | Persistent, metadata filtering, easy setup |
| Chunk size | 500 chars | Fits MiniLM's 256-token limit with headroom |
| LLM | Mistral-7B-Instruct | Strong instruction following, open-source |
| Sampling | Stratified by category | Balanced representation across all 4 products |

---

## KPI Targets

| KPI | Goal |
|-----|------|
| Trend identification time | Days → Minutes |
| Non-technical user access | No data analyst needed |
| Issue identification mode | Reactive → Proactive |

---

## Contributing

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Commit with conventional commits: `git commit -m "feat: add streaming support"`
3. Open a Pull Request to `main`

---

## License

Internal tool — CrediTrust Financial. Not for external distribution.
