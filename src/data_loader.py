import pyarrow.parquet as pq
import pandas as pd


def load_batch(path, columns=None):

    pf = pq.ParquetFile(path)

    table = pf.read(columns=columns)

    df = table.to_pandas()

    return df
