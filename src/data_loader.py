import pandas as pd

from pathlib import Path


def load_embeddings(path):

    df = pd.read_parquet(path)

    metadata = pd.json_normalize(df["metadata"])

    df = pd.concat(
        [
            df.drop(columns=["metadata"]),
            metadata
        ],
        axis=1
    )

    return df