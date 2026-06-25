from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_PATH = PROJECT_ROOT / "vector_store"

PARQUET_FILE = VECTOR_PATH / "complaint_embeddings.parquet"

FAISS_INDEX = VECTOR_PATH / "faiss.index"

TOP_K = 5

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "google/flan-t5-base"