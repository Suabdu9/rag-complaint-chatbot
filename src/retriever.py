import faiss
import numpy as np
import pyarrow.parquet as pq

from sentence_transformers import SentenceTransformer

from config import *


class Retriever:

    def __init__(self):

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.index = faiss.read_index(str(FAISS_INDEX))

        self.pf = pq.ParquetFile(PARQUET_FILE)

    def retrieve(self, query, k=TOP_K):

        q_emb = self.model.encode(query).astype("float32")

        _, idx = self.index.search(np.array([q_emb]), k)

        # fetch only needed rows
        table = self.pf.read_row_groups(
            self.pf.num_row_groups,
            columns=["document", "metadata"]
        )

        df = table.to_pandas()

        return df.iloc[idx[0]]
