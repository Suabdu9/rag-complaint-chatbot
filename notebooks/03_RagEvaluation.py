"""
Task 3: RAG Evaluation
Runs 10 representative questions through the pipeline and prints a scored table.
"""

import sys
sys.path.insert(0, ".")

from src.rag_pipeline import RAGPipeline

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation question set
# ──────────────────────────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "question": "Why are customers unhappy with their credit cards?",
        "product_filter": "Credit Card",
    },
    {
        "question": "What are the most common billing dispute issues with credit cards?",
        "product_filter": "Credit Card",
    },
    {
        "question": "What problems do customers face with personal loan repayments?",
        "product_filter": "Personal Loan",
    },
    {
        "question": "Are there fraud or identity theft complaints related to savings accounts?",
        "product_filter": "Savings Account",
    },
    {
        "question": "What issues do customers report when transferring money?",
        "product_filter": "Money Transfer",
    },
    {
        "question": "Which products generate the most complaints about customer service?",
        "product_filter": None,
    },
    {
        "question": "What are customers saying about unexpected fees or charges?",
        "product_filter": None,
    },
    {
        "question": "Are there patterns of complaints involving account closures?",
        "product_filter": None,
    },
    {
        "question": "What do customers say about delays in loan disbursement?",
        "product_filter": "Personal Loan",
    },
    {
        "question": "What are the most urgent compliance risks based on complaint narratives?",
        "product_filter": None,
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Run evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(pipeline: RAGPipeline) -> list[dict]:
    results = []
    for item in EVAL_QUESTIONS:
        print(f"\n{'='*70}")
        print(f"Q: {item['question']}")
        if item["product_filter"]:
            print(f"   [Filter: {item['product_filter']}]")

        result = pipeline.query(
            question=item["question"],
            product_filter=item.get("product_filter"),
        )

        print(f"\nA: {result['answer'][:500]} …")
        print(f"\nTop source snippet: {result['sources'][0]['text'][:200] if result['sources'] else 'N/A'} …")

        results.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "sources": result["sources"],
                "product_filter": item.get("product_filter"),
            }
        )

    return results


def print_eval_table(results: list[dict]) -> None:
    """Print a Markdown evaluation table for inclusion in the report."""
    print("\n\n## RAG Evaluation Results\n")
    print("| # | Question | Answer Summary | Top Source | Score (1–5) | Comments |")
    print("|---|----------|---------------|------------|-------------|----------|")
    for i, r in enumerate(results, 1):
        q = r["question"]
        ans = r["answer"][:120].replace("\n", " ") + "…"
        src = (
            r["sources"][0]["text"][:80].replace("\n", " ") + "…"
            if r["sources"]
            else "—"
        )
        print(f"| {i} | {q} | {ans} | {src} | _/5_ | |")

    print(
        "\n> **Note:** Fill in Score and Comments columns after manual review of each answer."
    )


if __name__ == "__main__":
    pipeline = RAGPipeline(backend="chroma")
    results = evaluate(pipeline)
    print_eval_table(results)
