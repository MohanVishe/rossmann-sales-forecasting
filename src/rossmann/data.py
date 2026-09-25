"""Load and validate the Kaggle Rossmann Store Sales files.

The data is not committed (see README, "Data"). `validate_files` checks the SHA-256 of
each file and `validate_frames` checks shape, date range and column stats against the
values the original 2023 notebook printed from Kaggle's own download.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# SHA-256 of the files as downloaded (identical across two independent Hugging Face mirrors).
EXPECTED_SHA256 = {
    "train.csv": "f6e4597c142d7d909a13d53b68a8e85c00b9a4c7b5ff40adbb37d6829cc1f4cc",
    "store.csv": "f56bd124a2849489e6bbb5c000f5fc9640204355e316475c918ae4d089afb344",
}

TRAIN_COLUMNS = [
    "Store", "DayOfWeek", "Date", "Sales", "Customers",
    "Open", "Promo", "StateHoliday", "SchoolHoliday",
]
STORE_COLUMNS = [
    "Store", "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
    "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
]

# Reference values printed by notebooks/original/individual-project.ipynb (Kaggle download).
EXPECTED_STATS = {
    "train_rows": 1_017_209,
    "store_rows": 1_115,
    "date_min": "2013-01-01",
    "date_max": "2015-07-31",
    "open_zero_rows": 172_817,
    "promo_rows": 388_080,
    "school_holiday_rows": 181_721,
    "competition_distance_missing": 3,
    "promo2_since_week_missing": 544,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_files(data_dir: Path) -> dict[str, str]:
    """Raise if a file is missing or its checksum differs; return the checksums."""
    found = {}
    for name, expected in EXPECTED_SHA256.items():
        path = Path(data_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - see README 'Data' for the download.")
        digest = sha256(path)
        if digest != expected:
            raise ValueError(f"{name}: sha256 {digest} != expected {expected}")
        found[name] = digest
    return found


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_csv(
        data_dir / "train.csv",
        dtype={"StateHoliday": str},
        parse_dates=["Date"],
    )
    store = pd.read_csv(data_dir / "store.csv")
    return train, store


def validate_frames(train: pd.DataFrame, store: pd.DataFrame) -> dict[str, object]:
    """Check shape and row stats against the reference values; return what was observed."""
    observed = {
        "train_rows": len(train),
        "store_rows": len(store),
        "date_min": train["Date"].min().strftime("%Y-%m-%d"),
        "date_max": train["Date"].max().strftime("%Y-%m-%d"),
        "open_zero_rows": int((train["Open"] == 0).sum()),
        "promo_rows": int((train["Promo"] == 1).sum()),
        "school_holiday_rows": int((train["SchoolHoliday"] == 1).sum()),
        "competition_distance_missing": int(store["CompetitionDistance"].isna().sum()),
        "promo2_since_week_missing": int(store["Promo2SinceWeek"].isna().sum()),
    }
    if list(train.columns) != TRAIN_COLUMNS:
        raise ValueError(f"train.csv columns {list(train.columns)} != {TRAIN_COLUMNS}")
    if list(store.columns) != STORE_COLUMNS:
        raise ValueError(f"store.csv columns {list(store.columns)} != {STORE_COLUMNS}")
    bad = {k: (observed[k], v) for k, v in EXPECTED_STATS.items() if observed[k] != v}
    if bad:
        raise ValueError(f"data does not match the Kaggle reference: {bad}")
    return observed
