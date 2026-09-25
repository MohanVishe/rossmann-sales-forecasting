"""Features that are known at forecast time.

Rules the tests enforce (tests/test_features.py):
- `Customers` is never an input. It is same-day footfall, unknown six weeks ahead and
  absent from Kaggle's test.csv.
- Every row-level feature for date d uses only rows dated <= d: store metadata, the
  calendar, and the promo/holiday schedule (which Kaggle's test.csv supplies for the
  forecast horizon).
- Target-derived store statistics come from `StoreEncoder`, fitted on the training
  window only.
- Stores that never joined Promo2 get Promo2 features of 0. No calendar is invented
  for them.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

HOLDOUT_DAYS = 42  # six weeks, the forecast horizon Rossmann's managers work to
RECENCY_CAP = 60  # days-since features are clipped here

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}
STORE_TYPE = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT = {"a": 0, "b": 1, "c": 2}
STATE_HOLIDAY = {"0": 0, "a": 1, "b": 2, "c": 3}

ROW_FEATURES = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday", "StateHolidayCode",
    "StoreTypeCode", "AssortmentCode", "Year", "Month", "Day", "WeekOfYear", "DayOfYear",
    "DaysToXmas", "CompetitionDistance", "CompetitionDistanceMissing",
    "CompetitionOpenMonths", "CompetitionActive", "CompetitionOpenUnknown",
    "Promo2Participant", "Promo2Active", "Promo2Weeks", "Promo2Month",
    "DaysSinceStateHoliday", "DaysSinceSchoolHoliday",
]
ENCODED_FEATURES = ["StoreLogMean", "StoreDowLogMean", "StorePromoLogMean"]
FEATURES = ROW_FEATURES + ENCODED_FEATURES


def time_split(df: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS):
    """Split on date: the last `holdout_days` days are the holdout."""
    last = df["Date"].max()
    cutoff = last - pd.Timedelta(days=holdout_days)  # last training date
    train = df[df["Date"] <= cutoff]
    holdout = df[df["Date"] > cutoff]
    return train, holdout, cutoff


def scoreable(df: pd.DataFrame) -> pd.DataFrame:
    """Open days with positive sales: the rows that are trained on and scored.

    Closed days have Sales = 0 by definition, and Kaggle's RMSPE ignores zero-sales days.
    """
    return df[(df["Open"] == 1) & (df["Sales"] > 0)]


def _promo2_start(store: pd.DataFrame) -> pd.Series:
    """Monday of ISO week Promo2SinceWeek/Promo2SinceYear; NaT for non-participants."""
    def start(row):
        if row["Promo2"] != 1 or pd.isna(row["Promo2SinceWeek"]):
            return pd.NaT
        return pd.Timestamp(dt.date.fromisocalendar(int(row["Promo2SinceYear"]),
                                                    int(row["Promo2SinceWeek"]), 1))
    return store.apply(start, axis=1)


def _competition_start(store: pd.DataFrame) -> pd.Series:
    year = store["CompetitionOpenSinceYear"]
    month = store["CompetitionOpenSinceMonth"]
    ok = year.notna() & month.notna()
    out = pd.Series(pd.NaT, index=store.index, dtype="datetime64[ns]")
    out[ok] = pd.to_datetime(dict(year=year[ok].astype(int), month=month[ok].astype(int), day=1))
    return out


def _days_since(df: pd.DataFrame, flag: pd.Series) -> pd.Series:
    """Days since the most recent flagged date for the same store, at or before each row."""
    event = df["Date"].where(flag)
    last = event.groupby(df["Store"]).ffill()  # rows are sorted by (Store, Date)
    return (df["Date"] - last).dt.days.fillna(RECENCY_CAP).clip(upper=RECENCY_CAP)


def row_features(sales: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Row-level features for every (Store, Date) row in `sales`.

    Pass all rows, including closed days, so the holiday-recency features see the full
    schedule. The result is sorted by (Store, Date) and keeps `Date`, `Sales` and `Open`
    for splitting and scoring; `Customers` is dropped.
    """
    df = sales.drop(columns=["Customers"], errors="ignore")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    meta = store.copy()
    meta["CompetitionStart"] = _competition_start(meta)
    meta["Promo2Start"] = _promo2_start(meta)
    df = df.merge(meta, on="Store", how="left", validate="many_to_one")

    date = df["Date"]
    df["Year"] = date.dt.year
    df["Month"] = date.dt.month
    df["Day"] = date.dt.day
    df["WeekOfYear"] = date.dt.isocalendar().week.astype(int).to_numpy()
    df["DayOfYear"] = date.dt.dayofyear
    xmas = pd.to_datetime(dict(year=df["Year"], month=12, day=25))
    df["DaysToXmas"] = (xmas - date).dt.days.where(lambda s: s >= 0, RECENCY_CAP).clip(upper=RECENCY_CAP)

    df["StoreTypeCode"] = df["StoreType"].map(STORE_TYPE)
    df["AssortmentCode"] = df["Assortment"].map(ASSORTMENT)
    df["StateHolidayCode"] = df["StateHoliday"].astype(str).map(STATE_HOLIDAY)

    # Competition: distance is a static snapshot; the start date tells us whether it applies yet.
    df["CompetitionDistanceMissing"] = df["CompetitionDistance"].isna().astype(int)
    months = ((df["Year"] - df["CompetitionStart"].dt.year) * 12
              + (df["Month"] - df["CompetitionStart"].dt.month))
    df["CompetitionOpenUnknown"] = (df["CompetitionStart"].isna()
                                    & df["CompetitionDistance"].notna()).astype(int)
    df["CompetitionActive"] = ((months >= 0) | (df["CompetitionOpenUnknown"] == 1)).astype(int)
    df["CompetitionOpenMonths"] = months.clip(lower=0)  # NaN when the start date is unknown

    # Promo2: only for stores that joined, and only from their start week.
    df["Promo2Participant"] = (df["Promo2"] == 1).astype(int)
    active = df["Promo2Start"].notna() & (date >= df["Promo2Start"])
    df["Promo2Active"] = active.astype(int)
    df["Promo2Weeks"] = np.where(active, (date - df["Promo2Start"]).dt.days // 7, 0)
    month_abbr = df["Month"].map(MONTH_ABBR)
    in_interval = [
        isinstance(interval, str) and abbr in interval.split(",")
        for interval, abbr in zip(df["PromoInterval"], month_abbr)
    ]
    df["Promo2Month"] = (active & pd.Series(in_interval, index=df.index)).astype(int)

    df["DaysSinceStateHoliday"] = _days_since(df, df["StateHolidayCode"] > 0)
    df["DaysSinceSchoolHoliday"] = _days_since(df, df["SchoolHoliday"] == 1)

    keep = ["Date", "Sales", "Open"] + ROW_FEATURES
    return df[[c for c in keep if c in df.columns]]


class StoreEncoder:
    """Per-store mean log sales (overall, by weekday, by promo), fitted on training rows only."""

    def fit(self, train: pd.DataFrame) -> "StoreEncoder":
        rows = scoreable(train)
        y = np.log1p(rows["Sales"])
        self.global_mean_ = float(y.mean())
        self.store_ = y.groupby(rows["Store"]).mean().rename("StoreLogMean")
        self.store_dow_ = y.groupby([rows["Store"], rows["DayOfWeek"]]).mean().rename("StoreDowLogMean")
        self.store_promo_ = y.groupby([rows["Store"], rows["Promo"]]).mean().rename("StorePromoLogMean")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.join(self.store_, on="Store")
        out = out.join(self.store_dow_, on=["Store", "DayOfWeek"])
        out = out.join(self.store_promo_, on=["Store", "Promo"])
        for col in ENCODED_FEATURES:
            out[col] = out[col].fillna(out["StoreLogMean"]).fillna(self.global_mean_)
        return out
