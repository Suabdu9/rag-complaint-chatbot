import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from src.config import *


class ComplaintRetriever:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.index = faiss.read_index(
            str(FAISS_INDEX)
        )

        self.metadata = pd.read_pickle(
            METADATA_FILE
        )

    def search(
        self,
        question,
        k=TOP_K
    ):

        query = self.model.encode(question)

        distances, indices = self.index.search(
            np.array([query]).astype("float32"),
            k
        )

        return self.metadata.iloc[
            indices[0]
        ]