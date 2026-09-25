"""Run the leak-free pipeline end to end and write results/metrics.json.

    uv run python scripts/run_pipeline.py            # expects data/train.csv and data/store.csv
    uv run python scripts/run_pipeline.py --data-dir path/to/data
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rossmann import data  # noqa: E402
from src.rossmann.features import (  # noqa: E402
    FEATURES, HOLDOUT_DAYS, StoreEncoder, row_features, scoreable, time_split,
)
from src.rossmann.models import (  # noqa: E402
    LinearModel, fit_lgb, same_weekday_last_year, score, store_weekday_median,
    store_weekday_promo_median,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "metrics.json")
    args = parser.parse_args()
    started = time.time()

    checksums = data.validate_files(args.data_dir)
    raw, store = data.load(args.data_dir)
    observed = data.validate_frames(raw, store)
    log(f"data validated: {observed['train_rows']:,} rows, {observed['store_rows']:,} stores")

    feats = row_features(raw, store)
    train_all, holdout_all, cutoff = time_split(feats)
    train, holdout = scoreable(train_all), scoreable(holdout_all)
    y_train, y_hold = train["Sales"].to_numpy(), holdout["Sales"].to_numpy()

    # Inner split inside the training window, used only to pick hyperparameters.
    inner_train_all, inner_val_all, inner_cutoff = time_split(train_all)
    inner_train, inner_val = scoreable(inner_train_all), scoreable(inner_val_all)

    results: dict[str, dict] = {}

    log("baselines")
    results["baseline_store_weekday_median"] = score(y_hold, store_weekday_median(train, holdout))
    results["baseline_store_weekday_promo_median"] = score(
        y_hold, store_weekday_promo_median(train, holdout))
    results["baseline_same_weekday_last_year"] = score(y_hold, same_weekday_last_year(train, holdout))

    # Store encodings: fitted on the inner training rows for model selection, then on the
    # full training window for the holdout. Never on holdout rows.
    enc_inner = StoreEncoder().fit(inner_train)
    Xi, Xv = enc_inner.transform(inner_train), enc_inner.transform(inner_val)
    enc = StoreEncoder().fit(train)
    Xt, Xh = enc.transform(train), enc.transform(holdout)

    log("linear: choose alpha on the inner validation window")
    alpha_scores = {}
    for alpha in (0.1, 1.0, 10.0):
        m = LinearModel(alpha).fit(Xi, np.log1p(Xi["Sales"]))
        alpha_scores[alpha] = score(Xv["Sales"], m.predict(Xv))["rmspe"]
    best_alpha = min(alpha_scores, key=alpha_scores.get)
    linear = LinearModel(best_alpha).fit(Xt, np.log1p(y_train))
    results["ridge_store_fixed_effects"] = score(y_hold, linear.predict(Xh))
    results["ridge_store_fixed_effects"]["alpha"] = best_alpha

    log("lightgbm: early-stop on the inner validation window")
    probe = fit_lgb(Xi, np.log1p(Xi["Sales"]), Xv, np.log1p(Xv["Sales"]))
    best_iter = probe.best_iteration
    inner_lgb = score(Xv["Sales"], np.expm1(probe.predict(Xv[FEATURES], num_iteration=best_iter)))

    log(f"lightgbm: refit on the full training window ({best_iter} rounds)")
    gbm = fit_lgb(Xt, np.log1p(y_train), num_boost_round=best_iter)
    results["lightgbm"] = score(y_hold, np.expm1(gbm.predict(Xh[FEATURES])))
    results["lightgbm"]["num_boost_round"] = best_iter
    importance = pd.Series(gbm.feature_importance("gain"), index=FEATURES)
    top_features = (importance / importance.sum()).sort_values(ascending=False).head(10).round(4)

    log("ablation: the same LightGBM with same-day Customers added (leaky, for comparison only)")
    cust = raw[["Store", "Date", "Customers"]]
    Xt_c = Xt.merge(cust, on=["Store", "Date"], how="left")
    Xh_c = Xh.merge(cust, on=["Store", "Date"], how="left")
    leaky_features = FEATURES + ["Customers"]
    gbm_leaky = fit_lgb(Xt_c, np.log1p(Xt_c["Sales"]), num_boost_round=best_iter,
                        features=leaky_features)
    results["ablation_lightgbm_with_same_day_customers_LEAKY"] = score(
        Xh_c["Sales"], np.expm1(gbm_leaky.predict(Xh_c[leaky_features])))

    out = {
        "generated_by": "scripts/run_pipeline.py",
        "data": {"sha256": checksums, "observed": observed},
        "split": {
            "holdout_days": HOLDOUT_DAYS,
            "train_dates": [str(train["Date"].min().date()), str(cutoff.date())],
            "holdout_dates": [str(holdout["Date"].min().date()), str(holdout["Date"].max().date())],
            "train_rows_scored": len(train),
            "holdout_rows_scored": len(holdout),
            "rows_scored": "Open == 1 and Sales > 0",
            "inner_validation_dates": [str(inner_val["Date"].min().date()),
                                       str(inner_val["Date"].max().date())],
        },
        "target": "log1p(Sales); metrics on the Sales scale (EUR)",
        "features": FEATURES,
        "holdout_metrics": results,
        "inner_validation": {"ridge_rmspe_by_alpha": {str(k): v for k, v in alpha_scores.items()},
                             "lightgbm": inner_lgb},
        "lightgbm_top_features_gain_share": top_features.to_dict(),
        "environment": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                        "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "runtime_seconds": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {args.out}")
    for name, m in results.items():
        print(f"  {name:52s} RMSPE {m['rmspe']:.4f}  R2 {m['r2']:.4f}  MAE {m['mae_eur']:.1f}")


if __name__ == "__main__":
    main()
