import json
from pathlib import Path

import pandas as pd

def load_decompositions_df(tweets_file_path: str | Path, decompositions_file_path: str | Path) -> pd.DataFrame:
    """One row per tweet, with a `decompositions` column holding that tweet's list of decompositions.

    The two files are line-aligned: line i of the decompositions file belongs to tweet i.
    """
    tweets_file_path = Path(tweets_file_path)
    decompositions_file_path = Path(decompositions_file_path)
    df = pd.read_json(tweets_file_path, lines=True, convert_dates=False,
                      dtype={"id": str, "user_id": str})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    with open(decompositions_file_path, encoding="utf-8") as f:
        decompositions = [json.loads(line) for line in f if line.strip()]
    if len(decompositions) != len(df):
        raise ValueError(f"{len(df)} tweets but {len(decompositions)} decompositions; "
                         "the files must be line-aligned")

    df["decompositions"] = decompositions
    return df
