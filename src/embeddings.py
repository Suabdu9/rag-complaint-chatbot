from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed(self, texts):

        return self.model.encode(
            texts,
            show_progress_bar=True
        )