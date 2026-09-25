"""Metrics, naive baselines, a linear model and a gradient-boosted tree model."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import ENCODED_FEATURES, FEATURES, scoreable

SEED = 42


def rmspe(y_true, y_pred) -> float:
    """Root mean square percentage error over rows with y_true > 0 (the Kaggle metric)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def score(y_true, y_pred) -> dict[str, float]:
    return {
        "rmspe": round(rmspe(y_true, y_pred), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae_eur": round(float(mean_absolute_error(y_true, y_pred)), 1),
    }


# ---------------------------------------------------------------- naive baselines

def store_weekday_median(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Median sales of the same store on the same weekday over the training window."""
    rows = scoreable(train)
    med = rows.groupby(["Store", "DayOfWeek"])["Sales"].median().rename("pred")
    return target.join(med, on=["Store", "DayOfWeek"])["pred"].to_numpy()


def store_weekday_promo_median(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Median sales of the same store, weekday and promo flag; falls back to store-weekday."""
    rows = scoreable(train)
    med = rows.groupby(["Store", "DayOfWeek", "Promo"])["Sales"].median().rename("pred")
    pred = target.join(med, on=["Store", "DayOfWeek", "Promo"])["pred"]
    return pred.fillna(pd.Series(store_weekday_median(train, target), index=target.index)).to_numpy()


def same_weekday_last_year(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Sales of the same store 364 days earlier (same weekday); falls back to store-weekday median."""
    rows = scoreable(train)[["Store", "Date", "Sales"]].copy()
    rows["Date"] = rows["Date"] + pd.Timedelta(days=364)
    lagged = target[["Store", "Date"]].merge(rows, on=["Store", "Date"], how="left")["Sales"]
    lagged.index = target.index
    fallback = pd.Series(store_weekday_median(train, target), index=target.index)
    return lagged.fillna(fallback).to_numpy()


# ---------------------------------------------------------------- linear model

LINEAR_CATEGORICAL = ["Store", "DayOfWeek", "Month", "StoreTypeCode", "AssortmentCode",
                      "StateHolidayCode"]
LINEAR_NUMERIC = ["Promo", "SchoolHoliday", "Promo2Active", "Promo2Month", "CompetitionActive",
                  "CompetitionDistanceMissing", "CompetitionOpenUnknown", "LogCompetitionDistance",
                  "CompetitionOpenMonthsCapped", "DaysToXmas", "DaysSinceStateHoliday",
                  "DaysSinceSchoolHoliday", "Year", "DayOfYear"] + ENCODED_FEATURES


def _linear_frame(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["LogCompetitionDistance"] = np.log1p(X["CompetitionDistance"].fillna(0))
    X["CompetitionOpenMonthsCapped"] = X["CompetitionOpenMonths"].fillna(0).clip(upper=120)
    return X


class LinearModel:
    """Ridge on log sales with store fixed effects (one-hot) and scaled numeric features.

    Store enters as a category, not a number. The train-fitted store encodings
    (store x weekday, store x promo mean log sales) supply the interactions a linear model
    cannot form itself. One-hot encoder and scaler are fitted on train only.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _matrix(self, X: pd.DataFrame, fit: bool):
        X = _linear_frame(X)
        if fit:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore").fit(X[LINEAR_CATEGORICAL])
            self.scaler_ = StandardScaler().fit(X[LINEAR_NUMERIC])
        cat = self.ohe_.transform(X[LINEAR_CATEGORICAL])
        num = sparse.csr_matrix(self.scaler_.transform(X[LINEAR_NUMERIC]))
        return sparse.hstack([cat, num]).tocsr()

    def fit(self, X: pd.DataFrame, y_log: np.ndarray) -> "LinearModel":
        self.model_ = Ridge(alpha=self.alpha, solver="sparse_cg", max_iter=5000, tol=1e-6)
        self.model_.fit(self._matrix(X, fit=True), y_log)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(self.model_.predict(self._matrix(X, fit=False)))


# ---------------------------------------------------------------- gradient-boosted trees

LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}


def fit_lgb(X: pd.DataFrame, y_log, X_val=None, y_val_log=None,
            num_boost_round: int = 4000, features=FEATURES):
    """Fit LightGBM on log sales. With a validation set, early-stop and record best_iteration."""
    train_set = lgb.Dataset(X[features], label=y_log, free_raw_data=False)
    if X_val is None:
        return lgb.train(LGB_PARAMS, train_set, num_boost_round=num_boost_round)
    val_set = lgb.Dataset(X_val[features], label=y_val_log, reference=train_set)
    return lgb.train(
        LGB_PARAMS, train_set, num_boost_round=num_boost_round, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
