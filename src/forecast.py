"""
forecast.py — Project FORESIGHT demand forecasting (D3)

- Seasonal-naive baseline (value from 7 days ago, i.e. same weekday last week)
- LightGBM model trained on lag/rolling/calendar features
- Rolling-origin backtest (multiple folds, never a random split)
- WAPE (primary) + MAPE + bias reported, model vs baseline
- Produces a forward forecast (with a simple quantile-based interval) per SKU
  for the next HORIZON weeks and saves it for the risk-scoring step.

Run: python src/forecast.py
Output:
  data/processed/backtest_results.json     (WAPE model vs baseline, per fold)
  data/processed/forecast_forward.parquet  (SKU-level forward forecast + interval)
  models/lgbm_forecast.pkl
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = Path(__file__).resolve().parent.parent / "models"
MODELS.mkdir(exist_ok=True)

HORIZON_DAYS = 56  # ~8 weeks, per brief's 6-8 week horizon
N_BACKTEST_FOLDS = 4
FOLD_TEST_DAYS = 28  # each fold tests on 4 weeks

FEATURES = [
    "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_28", "roll_std_28",
    "dow", "is_weekend", "week_of_year", "is_promo", "is_holiday",
    "sku_age_days", "unit_price",
]
CATEGORICAL = []  # kept numeric/simple; category handled via sku_age + separate global model


def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def bias(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(y_pred - y_true)


def seasonal_naive_predict(df, target_col="units_sold"):
    """Predict = value from 7 days ago for the same SKU (already computed as lag_7)."""
    return df["lag_7"].fillna(df["roll_mean_28"]).fillna(0)


def load_data():
    df = pd.read_parquet(PROCESSED / "analysis_ready.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["lag_28"])  # need full lag history to avoid leakage/NaN features
    return df.sort_values(["sku_id", "date"]).reset_index(drop=True)


def rolling_origin_backtest(df):
    """
    Rolling-origin CV: for each fold, train on everything before the fold's
    test window, test on the next FOLD_TEST_DAYS. Never shuffles time.
    """
    max_date = df["date"].max()
    results = []
    models_last_fold = None

    for fold in range(N_BACKTEST_FOLDS, 0, -1):
        test_end = max_date - pd.Timedelta(days=(fold - 1) * FOLD_TEST_DAYS)
        test_start = test_end - pd.Timedelta(days=FOLD_TEST_DAYS)
        train_end = test_start  # strictly before test window -> no leakage

        train = df[df["date"] < train_end]
        test = df[(df["date"] >= test_start) & (df["date"] < test_end)]

        if len(train) < 5000 or len(test) == 0:
            continue

        X_train, y_train = train[FEATURES], train["units_sold"]
        X_test, y_test = test[FEATURES], test["units_sold"]

        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        model_pred = np.clip(model.predict(X_test), 0, None)
        baseline_pred = seasonal_naive_predict(test)

        fold_result = {
            "fold": fold,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "n_test_rows": int(len(test)),
            "model_wape": round(float(wape(y_test, model_pred)), 4),
            "baseline_wape": round(float(wape(y_test, baseline_pred)), 4),
            "model_mape": round(float(mape(y_test, model_pred)), 4),
            "baseline_mape": round(float(mape(y_test, baseline_pred)), 4),
            "model_bias": round(float(bias(y_test, model_pred)), 4),
            "baseline_bias": round(float(bias(y_test, baseline_pred)), 4),
        }
        fold_result["model_beats_baseline"] = fold_result["model_wape"] < fold_result["baseline_wape"]
        results.append(fold_result)
        print(f"Fold {fold}: model WAPE={fold_result['model_wape']:.3f} "
              f"vs baseline WAPE={fold_result['baseline_wape']:.3f} "
              f"-> {'MODEL WINS' if fold_result['model_beats_baseline'] else 'baseline wins'}")

        models_last_fold = model  # keep the model trained on the most recent fold's data

    return results, models_last_fold


def train_final_model(df):
    """Train on ALL available history for the production forward forecast."""
    X, y = df[FEATURES], df["units_sold"]
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=30,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def build_forward_forecast(df, model):
    """
    Recursive multi-step forecast per SKU for HORIZON_DAYS ahead.
    Re-computes lag/rolling features at each step using prior predictions
    (standard approach for lag-feature models beyond the max lag).
    """
    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")

    forecasts = []
    for sku_id, g in df.groupby("sku_id"):
        g = g.sort_values("date")
        history = list(g["units_sold"].values[-60:])  # rolling buffer, most recent 60 days
        price = g["unit_price"].iloc[-1]
        launch_date = g["launch_date"].iloc[-1] if "launch_date" in g.columns else None
        sku_age_start = g["sku_age_days"].iloc[-1]

        preds = []
        for i, date in enumerate(future_dates):
            lag_7 = history[-7] if len(history) >= 7 else np.mean(history)
            lag_14 = history[-14] if len(history) >= 14 else np.mean(history)
            lag_28 = history[-28] if len(history) >= 28 else np.mean(history)
            roll_mean_7 = np.mean(history[-7:])
            roll_mean_28 = np.mean(history[-28:])
            roll_std_28 = np.std(history[-28:]) if len(history) >= 2 else 0.0

            row = {
                "lag_7": lag_7, "lag_14": lag_14, "lag_28": lag_28,
                "roll_mean_7": roll_mean_7, "roll_mean_28": roll_mean_28,
                "roll_std_28": roll_std_28,
                "dow": date.dayofweek, "is_weekend": int(date.dayofweek >= 5),
                "week_of_year": int(date.isocalendar().week),
                "is_promo": 0, "is_holiday": 0,
                "sku_age_days": sku_age_start + i + 1,
                "unit_price": price,
            }
            X_row = pd.DataFrame([row])[FEATURES]
            pred = max(float(model.predict(X_row)[0]), 0)
            preds.append(pred)
            history.append(pred)

        preds = np.array(preds)
        # simple uncertainty band from the SKU's recent residual volatility
        recent_std = g["units_sold"].tail(28).std()
        recent_std = 0.0 if pd.isna(recent_std) else recent_std
        lower = np.clip(preds - 1.28 * recent_std, 0, None)  # ~80% interval
        upper = preds + 1.28 * recent_std

        weekly_pred = preds.sum()
        forecasts.append({
            "sku_id": sku_id,
            "category": g["category"].iloc[-1],
            "forecast_horizon_days": HORIZON_DAYS,
            "forecast_total_units": round(float(weekly_pred), 1),
            "forecast_daily_mean": round(float(preds.mean()), 2),
            "forecast_lower_total": round(float(lower.sum()), 1),
            "forecast_upper_total": round(float(upper.sum()), 1),
            "unit_price": float(price),
        })

    return pd.DataFrame(forecasts)


def main():
    df = load_data()
    print(f"Modelling dataset: {df.shape}, date range {df['date'].min().date()} to {df['date'].max().date()}")

    print("\n--- Rolling-origin backtest (model vs seasonal-naive baseline) ---")
    fold_results, _ = rolling_origin_backtest(df)

    avg_model_wape = np.mean([f["model_wape"] for f in fold_results])
    avg_baseline_wape = np.mean([f["baseline_wape"] for f in fold_results])
    wins = sum(f["model_beats_baseline"] for f in fold_results)

    summary = {
        "n_folds": len(fold_results),
        "avg_model_wape": round(float(avg_model_wape), 4),
        "avg_baseline_wape": round(float(avg_baseline_wape), 4),
        "folds_model_won": f"{wins}/{len(fold_results)}",
        "model_beats_baseline_overall": bool(avg_model_wape < avg_baseline_wape),
        "folds": fold_results,
    }

    print(f"\nOverall: model avg WAPE={avg_model_wape:.4f} vs baseline avg WAPE={avg_baseline_wape:.4f}")
    print(f"Model beat baseline in {wins}/{len(fold_results)} folds.")
    print("Decision:", "SHIP THE MODEL" if summary["model_beats_baseline_overall"] else
                        "SHIP THE BASELINE (model did not win) — report this honestly")

    with open(PROCESSED / "backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining final model on full history for forward forecast...")
    final_model = train_final_model(df)
    with open(MODELS / "lgbm_forecast.pkl", "wb") as f:
        pickle.dump(final_model, f)

    print("Building forward forecast (recursive, per SKU)...")
    forward = build_forward_forecast(df, final_model)
    forward.to_parquet(PROCESSED / "forecast_forward.parquet", index=False)
    print(f"Saved forward forecast for {len(forward)} SKUs -> {PROCESSED / 'forecast_forward.parquet'}")


if __name__ == "__main__":
    main()
