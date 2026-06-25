import faiss
import numpy as np
import pyarrow.parquet as pq
from config import PARQUET_FILE, FAISS_INDEX


def main():

    print("Reading parquet (streaming embeddings only)...")

    pf = pq.ParquetFile(PARQUET_FILE)

    table = pf.read(columns=["embedding"])

    # MUCH faster than to_pylist()
    embeddings = np.vstack(
        table.column("embedding").to_numpy()
    ).astype("float32")

    print("Shape:", embeddings.shape)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    print("Building FAISS index... (this is the slow step)")

    index.add(embeddings)

    print("Saving index...")

    faiss.write_index(index, str(FAISS_INDEX))

    print("DONE ✔")


if __name__ == "__main__":
    main()
