"""
app.py — Task 4: CrediTrust Complaint Intelligence Chatbot
Gradio UI wrapping the RAG pipeline (src/rag_pipeline.py)

Run with:
    python app.py
or
    gradio app.py
"""

import os
import gradio as gr
from src.rag_pipeline import RAGPipeline

# ──────────────────────────────────────────────────────────────────────────────
# Initialise pipeline (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────────────

BACKEND = os.getenv("VECTOR_STORE_TYPE", "chroma")
pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline(backend=BACKEND)
    return pipeline


PRODUCT_CHOICES = [
    "All Products",
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
]

EXAMPLE_QUESTIONS = [
    "Why are customers unhappy with credit card billing?",
    "What are the most common money transfer problems?",
    "Are there fraud complaints related to savings accounts?",
    "What do customers say about loan repayment issues?",
    "Which issues appear most urgent for compliance teams?",
]


# ──────────────────────────────────────────────────────────────────────────────
# Core query handler
# ──────────────────────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    product_filter: str,
    top_k: int,
    history: list[list[str]],
) -> tuple[list[list[str]], str]:
    """
    Called by Gradio on every Submit click.
    Returns updated chat history and formatted source citations.
    """
    if not question.strip():
        return history, ""

    product = None if product_filter == "All Products" else product_filter

    try:
        rag = get_pipeline()
        result = rag.query(question=question, k=top_k, product_filter=product)
        answer = result["answer"]
        sources = result["sources"]
    except Exception as e:
        answer = f"⚠️ Error running query: {str(e)}"
        sources = []

    # Append to chat history
    history = history + [[question, answer]]

    # Format source citations panel
    source_md = _format_sources(sources)

    return history, source_md


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_No sources retrieved._"

    lines = ["### 📄 Retrieved Source Chunks\n"]
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata", {})
        score = s.get("score", 0.0)
        category = meta.get("product_category", "Unknown")
        issue = meta.get("issue", "—")
        date = meta.get("date_received", "—")
        company = meta.get("company", "—")
        text_preview = s["text"][:300].replace("\n", " ")

        lines.append(
            f"**Source {i}** | 🏷️ {category} | 📋 Issue: {issue} | "
            f"🏢 {company} | 📅 {date} | Score: {score:.3f}\n"
            f"> {text_preview}…\n"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
#chatbot { height: 520px; overflow-y: auto; }
#sources { max-height: 400px; overflow-y: auto; font-size: 0.85em; }
.header-logo { text-align: center; padding: 8px 0; }
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CSS,
    title="CrediTrust Complaint Intelligence",
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 🏦 CrediTrust Complaint Intelligence
        ### Powered by RAG · Ask questions about customer complaints across all products
        """
    )

    with gr.Row():
        # ── Left: Chat ──────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Conversation",
                bubble_full_width=False,
                show_label=False,
            )

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="e.g. Why are credit card customers complaining about fees?",
                    show_label=False,
                    scale=5,
                    lines=2,
                    max_lines=4,
                )
                submit_btn = gr.Button("Ask ✈️", variant="primary", scale=1, min_width=80)

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=question_box,
                    label="Example questions",
                )

        # ── Right: Controls + Sources ────────────────────────────────────────
        with gr.Column(scale=2):
            product_filter = gr.Dropdown(
                choices=PRODUCT_CHOICES,
                value="All Products",
                label="🔍 Filter by Product",
            )
            top_k_slider = gr.Slider(
                minimum=3,
                maximum=15,
                value=5,
                step=1,
                label="📚 Number of source chunks (top-k)",
            )

            gr.Markdown("---")

            sources_display = gr.Markdown(
                value="_Sources will appear here after your first question._",
                elem_id="sources",
                label="Retrieved Sources",
            )

    # ── State ───────────────────────────────────────────────────────────────
    chat_history = gr.State([])

    # ── Event handlers ───────────────────────────────────────────────────────
    def on_submit(question, product, top_k, history):
        new_history, sources_md = answer_question(question, product, top_k, history)
        return new_history, new_history, sources_md, ""

    submit_btn.click(
        fn=on_submit,
        inputs=[question_box, product_filter, top_k_slider, chat_history],
        outputs=[chatbot, chat_history, sources_display, question_box],
    )

    question_box.submit(
        fn=on_submit,
        inputs=[question_box, product_filter, top_k_slider, chat_history],
        outputs=[chatbot, chat_history, sources_display, question_box],
    )

    def on_clear():
        return [], [], "_Sources will appear here after your first question._"

    clear_btn.click(
        fn=on_clear,
        inputs=[],
        outputs=[chatbot, chat_history, sources_display],
    )

    gr.Markdown(
        "_CrediTrust Financial · Internal Tool · Answers are grounded in retrieved "
        "complaint data. Always verify before acting on insights._",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
    )
