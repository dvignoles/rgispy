import gzip
from io import StringIO

import pandas as pd


def read_tsv_raw(tsv_path):
    if tsv_path.name.endswith(".gz"):
        with gzip.open(tsv_path, "rt") as f:
            return f.read()
    else:
        with open(tsv_path, "r") as f:
            return f.read()


def read_tsv(tsv_path):
    df = pd.read_csv(StringIO(read_tsv_raw(tsv_path)), delimiter="\t")
    return df
