import json
from pathlib import Path

import pandas as pd

# Time zone of the raw timestamps (offsets -04:00 / -05:00).
TIMEZONE = "America/New_York"


def to_created_at_day(datetime_utc: pd.Series) -> pd.Series:
    """Calendar day ("YYYY-MM-DD") of UTC timestamps in US Eastern time.

    This is how `created_at_day` is defined in the annotation data the graph construction is
    built on, while `datetime` itself stays in UTC. Use it wherever tweets are grouped by day.
    """
    return datetime_utc.dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d")


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


def load_annotations_df(annotations_file_path: str | Path, tweets_file_path: str | Path,
                        decompositions_file_path: str | Path) -> pd.DataFrame:
    """Rebuild the full annotation table from a stripped annotation CSV and the raw data.

    The annotation CSVs (data/annotations/) only hold `id` and our annotations. Tweet text,
    decompositions and the metadata of Hoyle et al. are added back from the raw files by `id`.
    Rows keep the order of the annotation file.
    """
    annotations = pd.read_csv(annotations_file_path, dtype={"id": str})
    df = load_decompositions_df(tweets_file_path, decompositions_file_path)

    missing = ~annotations["id"].isin(df["id"])
    if missing.any():
        raise ValueError(f"{missing.sum()} annotated ids are not in {Path(tweets_file_path).name}, "
                         f"e.g. {annotations.loc[missing, 'id'].head(3).tolist()}")

    df["created_at_day"] = to_created_at_day(df["datetime"])

    return df.merge(annotations, on="id", how="right", validate="one_to_one")
