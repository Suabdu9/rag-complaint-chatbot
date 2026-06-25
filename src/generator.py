from transformers import pipeline
from config import LLM_MODEL
from prompts import RAG_PROMPT


class Generator:

    def __init__(self):

        self.pipe = pipeline(
            "text-generation",
            model=LLM_MODEL,
            max_new_tokens=256
        )

    def generate(self, question, docs):

        context = "\n\n".join(docs["document"].tolist())

        prompt = RAG_PROMPT.format(
            context=context,
            question=question
        )

        return self.pipe(prompt)[0]["generated_text"]
