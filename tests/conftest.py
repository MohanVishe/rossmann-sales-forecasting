"""Synthetic data shaped like the Kaggle files, so tests run without the real download."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def store() -> pd.DataFrame:
    return pd.DataFrame({
        "Store": [1, 2, 3, 4],
        "StoreType": ["a", "b", "c", "d"],
        "Assortment": ["a", "b", "c", "a"],
        "CompetitionDistance": [1270.0, 570.0, np.nan, 300.0],
        "CompetitionOpenSinceMonth": [9.0, np.nan, np.nan, 3.0],
        "CompetitionOpenSinceYear": [2008.0, np.nan, np.nan, 2014.0],
        "Promo2": [0, 1, 0, 1],
        "Promo2SinceWeek": [np.nan, 13.0, np.nan, 40.0],
        "Promo2SinceYear": [np.nan, 2013.0, np.nan, 2014.0],
        "PromoInterval": [np.nan, "Jan,Apr,Jul,Oct", np.nan, "Mar,Jun,Sept,Dec"],
    })


@pytest.fixture
def sales() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2013-01-01", "2015-07-31", freq="D")
    frames = []
    for s in [1, 2, 3, 4]:
        n = len(dates)
        open_ = (dates.dayofweek != 6).astype(int)
        sales = np.where(open_ == 1, rng.integers(2000, 9000, n), 0)
        holiday = rng.choice(["0", "0", "0", "0", "0", "0", "0", "0", "a", "b"], n)
        frames.append(pd.DataFrame({
            "Store": s,
            "DayOfWeek": dates.dayofweek + 1,
            "Date": dates,
            "Sales": sales,
            "Customers": np.where(open_ == 1, sales // 9, 0),
            "Open": open_,
            "Promo": rng.integers(0, 2, n),
            "StateHoliday": holiday,
            "SchoolHoliday": rng.integers(0, 2, n),
        }))
    return pd.concat(frames, ignore_index=True)
