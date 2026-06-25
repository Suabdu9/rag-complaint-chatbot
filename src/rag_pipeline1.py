"""
src/rag_pipeline.py
Task 3: RAG Core Logic — Retriever + Prompt Engineering + Generator

Supports two vector backend options:
  - ChromaDB  (default, persistent)
  - FAISS     (set USE_FAISS=True in env or pass backend="faiss")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL_ID = os.getenv(
    "EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL_ID = os.getenv(
    "LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"
)
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "vector_store/chroma"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "vector_store/faiss_index.bin"))
FAISS_META_PATH = Path(os.getenv("FAISS_META_PATH", "vector_store/faiss_metadata.json"))
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a senior financial analyst assistant at CrediTrust Financial.
Your role is to help internal teams (Product, Support, Compliance) understand customer complaints.

Instructions:
- Answer the question using ONLY the provided complaint excerpts below.
- Be specific: cite patterns, recurring themes, and notable examples from the context.
- If the context does not contain enough information to answer fully, say so clearly.
- Do NOT fabricate information or draw on knowledge outside the provided context.
- Format your answer in clear, concise prose. Use bullet points only for lists of distinct issues.

---
Retrieved Complaint Excerpts:
{context}
---

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Combine question and retrieved chunks into a grounded prompt."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        category = meta.get("product_category", "Unknown")
        issue = meta.get("issue", "")
        date = meta.get("date_received", "")
        header = f"[{i}] Product: {category} | Issue: {issue} | Date: {date}"
        context_parts.append(f"{header}\n{chunk['text']}")

    context = "\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)


# ──────────────────────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────────────────────

class ChromaRetriever:
    """Semantic search over a ChromaDB collection."""

    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_collection("complaints")
        self._model = SentenceTransformer(EMBEDDING_MODEL_ID)
        print(
            f"[ChromaRetriever] Loaded collection with "
            f"{self._collection.count():,} chunks."
        )

    def retrieve(
        self,
        question: str,
        k: int = TOP_K,
        product_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Embed the question and return the top-k most similar chunks.

        Args:
            question: Natural-language user query.
            k:        Number of chunks to retrieve.
            product_filter: Optional product category to restrict results.
        """
        query_embedding = self._model.encode(
            [question], normalize_embeddings=True
        ).tolist()

        where = (
            {"product_category": product_filter} if product_filter else None
        )

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": text,
                    "metadata": meta,
                    "score": float(1 - dist),  # cosine similarity
                }
            )
        return chunks


class FAISSRetriever:
    """Semantic search using a FAISS flat index."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_META_PATH,
    ):
        import faiss

        self._index = faiss.read_index(str(index_path))
        with open(meta_path) as f:
            data = json.load(f)
        self._texts = data["texts"]
        self._metadata = data["metadata"]
        self._model = SentenceTransformer(EMBEDDING_MODEL_ID)
        print(f"[FAISSRetriever] Loaded index with {self._index.ntotal:,} vectors.")

    def retrieve(
        self,
        question: str,
        k: int = TOP_K,
        product_filter: Optional[str] = None,
    ) -> list[dict]:
        query_vec = self._model.encode(
            [question], normalize_embeddings=True
        ).astype(np.float32)

        # Retrieve more candidates if we'll filter by product
        fetch_k = k * 5 if product_filter else k
        distances, indices = self._index.search(query_vec, fetch_k)

        chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            if product_filter and meta.get("product_category") != product_filter:
                continue
            chunks.append(
                {
                    "text": self._texts[idx],
                    "metadata": meta,
                    "score": float(dist),
                }
            )
            if len(chunks) >= k:
                break

        return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Generator (LLM)
# ──────────────────────────────────────────────────────────────────────────────

class LLMGenerator:
    """
    Wraps a Hugging Face text-generation pipeline.
    On first call, downloads the model (may take time on first run).
    """

    def __init__(self, model_id: str = LLM_MODEL_ID):
        from transformers import pipeline

        print(f"[LLMGenerator] Loading model: {model_id} …")
        self._pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            torch_dtype="auto",
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        self._model_id = model_id

    def generate(self, prompt: str) -> str:
        """Run the prompt through the LLM and return the generated text."""
        outputs = self._pipe(prompt, return_full_text=False)
        return outputs[0]["generated_text"].strip()


# ──────────────────────────────────────────────────────────────────────────────
# RAG Pipeline (orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG: retrieve relevant chunks → build grounded prompt → generate answer.
    """

    def __init__(self, backend: str = "chroma"):
        if backend == "faiss":
            self.retriever = FAISSRetriever()
        else:
            self.retriever = ChromaRetriever()

        self.generator = LLMGenerator()

    def query(
        self,
        question: str,
        k: int = TOP_K,
        product_filter: Optional[str] = None,
    ) -> dict:
        """
        Args:
            question:       User's natural-language question.
            k:              Number of chunks to retrieve.
            product_filter: Optional product category filter.

        Returns:
            dict with keys: answer, sources, question
        """
        # Step 1 — Retrieve
        chunks = self.retriever.retrieve(question, k=k, product_filter=product_filter)

        # Step 2 — Build grounded prompt
        prompt = build_prompt(question, chunks)

        # Step 3 — Generate
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": chunks,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = RAGPipeline(backend="chroma")
    result = pipeline.query("Why are customers unhappy with credit card billing?")
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== TOP SOURCES ===")
    for s in result["sources"][:2]:
        print(f"  [{s['score']:.3f}] {s['text'][:200]} …")
