from retriever import Retriever
from generator import Generator


class RAG:

    def __init__(self):

        self.retriever = Retriever()
        self.generator = Generator()

    def ask(self, question):

        docs = self.retriever.retrieve(question)

        answer = self.generator.generate(question, docs)

        return {
            "answer": answer,
            "sources": docs
        }
