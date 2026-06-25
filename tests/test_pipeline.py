"""
Unit tests for the RAG pipeline components.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────────────────────────────────────
# Tests: build_prompt
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_prompt_contains_question(self):
        from src.rag_pipeline import build_prompt

        chunks = [
            {
                "text": "Customer complained about late fees.",
                "metadata": {"product_category": "Credit Card", "issue": "Billing", "date_received": "2023-01-01"},
                "score": 0.9,
            }
        ]
        prompt = build_prompt("Why are fees high?", chunks)
        assert "Why are fees high?" in prompt

    def test_prompt_contains_context(self):
        from src.rag_pipeline import build_prompt

        chunks = [
            {
                "text": "Unexpected charges on my statement.",
                "metadata": {"product_category": "Credit Card", "issue": "Fees", "date_received": "2023-02-01"},
                "score": 0.85,
            }
        ]
        prompt = build_prompt("What billing issues exist?", chunks)
        assert "Unexpected charges on my statement." in prompt

    def test_prompt_contains_instructions(self):
        from src.rag_pipeline import build_prompt

        prompt = build_prompt("Test question?", [])
        # Should contain the system instructions
        assert "CrediTrust" in prompt
        assert "Answer:" in prompt

    def test_prompt_multiple_chunks(self):
        from src.rag_pipeline import build_prompt

        chunks = [
            {
                "text": f"Complaint text {i}",
                "metadata": {"product_category": "Personal Loan", "issue": "Payment", "date_received": "2023-01-01"},
                "score": 0.9 - i * 0.1,
            }
            for i in range(3)
        ]
        prompt = build_prompt("Loan question?", chunks)
        for i in range(3):
            assert f"Complaint text {i}" in prompt


# ──────────────────────────────────────────────────────────────────────────────
# Tests: text cleaning (from Task 1)
# ──────────────────────────────────────────────────────────────────────────────

class TestCleanText:
    def setup_method(self):
        # Import dynamically to avoid heavy imports at collection time
        import sys, importlib
        sys.path.insert(0, ".")
        # We'll import the function directly
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "task1", "notebooks/task1_eda_preprocessing.py"
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_lowercases_text(self):
        result = self.mod.clean_text("BILLING DISPUTE")
        assert result == result.lower()

    def test_removes_boilerplate(self):
        text = "I am writing to file a complaint about my credit card."
        result = self.mod.clean_text(text)
        # boilerplate phrase should be gone
        assert "i am writing to file a complaint" not in result

    def test_collapses_whitespace(self):
        result = self.mod.clean_text("too   many    spaces")
        assert "  " not in result

    def test_handles_empty_string(self):
        assert self.mod.clean_text("") == ""

    def test_handles_none(self):
        assert self.mod.clean_text(None) == ""

    def test_removes_special_chars(self):
        result = self.mod.clean_text("Hello @World! #test $$$")
        # Should not contain @ # $
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result


# ──────────────────────────────────────────────────────────────────────────────
# Tests: RAGPipeline (mocked — no real model needed for unit tests)
# ──────────────────────────────────────────────────────────────────────────────

class TestRAGPipelineMocked:
    def test_query_returns_expected_keys(self):
        from src.rag_pipeline import build_prompt

        # We only test build_prompt directly since full pipeline needs GPU/models
        chunks = [
            {
                "text": "Sample complaint about fees.",
                "metadata": {"product_category": "Credit Card", "issue": "Fees", "date_received": "2023-01-01"},
                "score": 0.9,
            }
        ]
        result = build_prompt("Fee question?", chunks)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_empty_chunks_prompt(self):
        from src.rag_pipeline import build_prompt
        # Should not raise even with empty chunk list
        prompt = build_prompt("Any question?", [])
        assert "Any question?" in prompt

    def test_prompt_template_structure(self):
        from src.rag_pipeline import PROMPT_TEMPLATE
        # Ensure template has required placeholders
        assert "{context}" in PROMPT_TEMPLATE
        assert "{question}" in PROMPT_TEMPLATE
