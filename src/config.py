from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_STORE = PROJECT_ROOT / "vector_store"

PARQUET_FILE = VECTOR_STORE / "complaint_embeddings.parquet"

FAISS_INDEX = VECTOR_STORE / "faiss.index"

METADATA_FILE = VECTOR_STORE / "metadata.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

LLM_NAME = "google/flan-t5-base"
