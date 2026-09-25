import numpy as np
import pandas as pd
import pytest

from src.rossmann import data
from src.rossmann.features import (
    ENCODED_FEATURES, FEATURES, HOLDOUT_DAYS, ROW_FEATURES, StoreEncoder, row_features,
    scoreable, time_split,
)
from src.rossmann.models import rmspe, same_weekday_last_year, store_weekday_median


def _perturb_after(sales: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    """Scramble every column of every row dated after `cutoff`."""
    rng = np.random.default_rng(seed)
    out = sales.copy()
    later = out["Date"] > cutoff
    n = int(later.sum())
    out.loc[later, "Sales"] = rng.integers(0, 50_000, n)
    out.loc[later, "Customers"] = rng.integers(0, 5_000, n)
    out.loc[later, "Open"] = rng.integers(0, 2, n)
    out.loc[later, "Promo"] = rng.integers(0, 2, n)
    out.loc[later, "SchoolHoliday"] = rng.integers(0, 2, n)
    out.loc[later, "StateHoliday"] = rng.choice(["0", "a", "b", "c"], n)
    return out


# ---------------------------------------------------------------- Customers

def test_customers_is_not_a_feature():
    assert "Customers" not in FEATURES


def test_customers_is_dropped_and_has_no_effect(sales, store):
    base = row_features(sales, store)
    assert "Customers" not in base.columns
    changed = sales.assign(Customers=sales["Customers"][::-1].to_numpy() * 7)
    pd.testing.assert_frame_equal(base, row_features(changed, store))


def test_encoder_ignores_customers(sales, store):
    feats = row_features(sales, store)
    enc = StoreEncoder().fit(feats)
    assert not any("Customer" in c for c in enc.transform(feats).columns)


# ---------------------------------------------------------------- no future leakage

@pytest.mark.parametrize("cutoff", ["2013-03-15", "2014-06-30", "2015-06-19", "2015-07-30"])
def test_row_features_use_only_data_up_to_each_date(sales, store, cutoff):
    """Features for date d must not change when anything after d changes."""
    cutoff = pd.Timestamp(cutoff)
    before = row_features(sales, store)
    after = row_features(_perturb_after(sales, cutoff), store)
    keep = before["Date"] <= cutoff
    pd.testing.assert_frame_equal(
        before.loc[keep, ROW_FEATURES].reset_index(drop=True),
        after.loc[keep, ROW_FEATURES].reset_index(drop=True),
    )


def test_encoder_fitted_on_train_is_blind_to_holdout(sales, store):
    feats = row_features(sales, store)
    train, holdout, cutoff = time_split(feats)
    enc = StoreEncoder().fit(train)
    base = enc.transform(holdout)[ENCODED_FEATURES]

    scrambled = row_features(_perturb_after(sales, cutoff), store)
    train2, holdout2, _ = time_split(scrambled)
    enc2 = StoreEncoder().fit(train2)
    # Same schedule columns so the lookup keys match; only holdout targets differ.
    holdout2 = holdout2.assign(DayOfWeek=holdout["DayOfWeek"].to_numpy(),
                               Promo=holdout["Promo"].to_numpy())
    pd.testing.assert_frame_equal(base.reset_index(drop=True),
                                  enc2.transform(holdout2)[ENCODED_FEATURES].reset_index(drop=True))


def test_encoder_values_come_from_training_rows_only(sales, store):
    feats = row_features(sales, store)
    train, _, _ = time_split(feats)
    enc = StoreEncoder().fit(train)
    expected = np.log1p(scoreable(train).query("Store == 2")["Sales"]).mean()
    assert enc.store_.loc[2] == pytest.approx(expected)


def test_baselines_use_training_window_only(sales, store):
    feats = row_features(sales, store)
    train, holdout, cutoff = time_split(feats)
    target = scoreable(holdout)
    base = same_weekday_last_year(train, target)
    scrambled_train, _, _ = time_split(row_features(_perturb_after(sales, cutoff), store))
    np.testing.assert_array_equal(base, same_weekday_last_year(scrambled_train, target))
    np.testing.assert_array_equal(store_weekday_median(train, target),
                                  store_weekday_median(scrambled_train, target))


# ---------------------------------------------------------------- split and scoring

def test_time_split_holds_out_last_six_weeks(sales, store):
    feats = row_features(sales, store)
    train, holdout, cutoff = time_split(feats)
    assert HOLDOUT_DAYS == 42
    assert cutoff == pd.Timestamp("2015-06-19")
    assert holdout["Date"].min() == pd.Timestamp("2015-06-20")
    assert holdout["Date"].nunique() == 42
    assert train["Date"].max() < holdout["Date"].min()


def test_scoreable_drops_closed_and_zero_sales_days():
    df = pd.DataFrame({"Open": [1, 1, 0, 1], "Sales": [10, 0, 0, 5]})
    assert scoreable(df).index.tolist() == [0, 3]


def test_rmspe_matches_hand_calculation_and_ignores_zero_sales():
    assert rmspe([100, 200, 0], [110, 180, 50]) == pytest.approx(np.sqrt((0.01 + 0.01) / 2))


# ---------------------------------------------------------------- Promo2 and competition timing

def _row(feats, store_id, date):
    return feats[(feats["Store"] == store_id) & (feats["Date"] == pd.Timestamp(date))].iloc[0]


def test_non_participants_get_no_promo2_calendar(sales, store):
    feats = row_features(sales, store)
    non = feats[feats["Store"].isin([1, 3])]
    assert (non[["Promo2Participant", "Promo2Active", "Promo2Weeks", "Promo2Month"]] == 0).all().all()


def test_promo2_starts_on_its_iso_week(sales, store):
    feats = row_features(sales, store)
    # Store 2 joined in ISO week 13 of 2013, which starts Monday 2013-03-25.
    assert _row(feats, 2, "2013-03-24")["Promo2Active"] == 0
    assert _row(feats, 2, "2013-03-25")["Promo2Active"] == 1
    assert _row(feats, 2, "2013-04-01")["Promo2Weeks"] == 1
    assert _row(feats, 2, "2013-04-10")["Promo2Month"] == 1  # April is in its interval
    assert _row(feats, 2, "2013-05-10")["Promo2Month"] == 0
    # Store 4's interval spells September "Sept".
    assert _row(feats, 4, "2014-09-28")["Promo2Month"] == 0  # the day before ISO week 40
    assert _row(feats, 4, "2014-09-29")["Promo2Month"] == 1  # first day of its Promo2
    assert _row(feats, 4, "2015-06-10")["Promo2Month"] == 1


def test_competition_timing(sales, store):
    feats = row_features(sales, store)
    # Store 4's competitor opened March 2014.
    assert _row(feats, 4, "2014-02-28")["CompetitionActive"] == 0
    assert _row(feats, 4, "2014-03-01")["CompetitionActive"] == 1
    assert _row(feats, 4, "2015-03-01")["CompetitionOpenMonths"] == 12
    # Store 2 has a distance but no start date: flagged unknown, months left missing.
    r = _row(feats, 2, "2014-01-01")
    assert r["CompetitionOpenUnknown"] == 1 and np.isnan(r["CompetitionOpenMonths"])
    # Store 3 has no competitor on record.
    assert _row(feats, 3, "2014-01-01")["CompetitionDistanceMissing"] == 1


def test_days_since_holiday_counts_back_within_store(store):
    dates = pd.date_range("2015-01-01", periods=6)
    sales = pd.DataFrame({
        "Store": 1, "DayOfWeek": dates.dayofweek + 1, "Date": dates, "Sales": 100,
        "Customers": 10, "Open": 1, "Promo": 0,
        "StateHoliday": ["a", "0", "0", "b", "0", "0"], "SchoolHoliday": 0,
    })
    feats = row_features(sales, store)
    assert feats["DaysSinceStateHoliday"].tolist() == [0, 1, 2, 0, 1, 2]


# ---------------------------------------------------------------- data validation

def test_validate_frames_rejects_wrong_shape(sales, store):
    with pytest.raises(ValueError, match="Kaggle reference"):
        data.validate_frames(sales, store)
