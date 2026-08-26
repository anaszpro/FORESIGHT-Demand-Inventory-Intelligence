"""
risk.py — Project FORESIGHT stockout / overstock risk scoring (D4)

Combines the forward demand forecast with current inventory position to
score, for every SKU:
  - stockout_risk (0-1): projected stock over lead time vs demand over lead time
  - overstock_risk (0-1): on-hand stock vs demand over a forward window
  - quadrant: Reorder Now / Markdown-Clear / Watch-Volatile / Healthy
  - recommended_action
  - rupee value at stake (sales at risk / capital locked)

Transparent, rule-based scoring (not a black box) per the brief.

Run: python src/risk.py
Output: data/processed/risk_scored.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

OVERSTOCK_WINDOW_DAYS = 56  # compare on-hand vs demand over this many days
STOCKOUT_SAFETY_MULT = 1.0  # safety buffer multiplier over lead-time demand


def load_inputs():
    forecast = pd.read_parquet(PROCESSED / "forecast_forward.parquet")
    inventory = pd.read_parquet(PROCESSED / "inventory_clean.parquet")
    sku_master = pd.read_parquet(PROCESSED / "sku_master_clean.parquet")

    inventory["date"] = pd.to_datetime(inventory["date"])
    latest_inv = (
        inventory.sort_values("date")
        .groupby("sku_id")
        .tail(1)
        .set_index("sku_id")
    )
    return forecast, latest_inv, sku_master


def score_risk(forecast: pd.DataFrame, latest_inv: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    df = forecast.merge(latest_inv, on="sku_id", how="left")
    df = df.merge(sku_master[["sku_id", "unit_cost"]], on="sku_id", how="left")

    horizon_days = df["forecast_horizon_days"].iloc[0]
    daily_demand = df["forecast_daily_mean"]

    # --- stockout risk ---
    # demand expected over the SKU's own lead time, with a safety buffer
    df["lead_time_days"] = df["lead_time_days"].fillna(14)
    df["demand_over_lead_time"] = daily_demand * df["lead_time_days"] * STOCKOUT_SAFETY_MULT
    df["projected_position"] = df["on_hand_units"].fillna(0) + df["on_order_units"].fillna(0)

    # risk score: how far short projected position is vs demand-over-lead-time, scaled 0-1
    shortfall = (df["demand_over_lead_time"] - df["projected_position"]) / df["demand_over_lead_time"].replace(0, np.nan)
    df["stockout_risk"] = shortfall.clip(0, 1).fillna(0)

    # --- overstock risk ---
    demand_over_window = daily_demand * OVERSTOCK_WINDOW_DAYS
    excess = (df["on_hand_units"].fillna(0) - demand_over_window) / demand_over_window.replace(0, np.nan)
    df["overstock_risk"] = excess.clip(0, 1).fillna(0)

    # --- quadrant classification ---
    def classify(row):
        so, ov = row["stockout_risk"], row["overstock_risk"]
        if so >= 0.5 and ov >= 0.5:
            return "Watch / Volatile"
        elif so >= 0.5:
            return "Reorder Now"
        elif ov >= 0.5:
            return "Markdown / Clear"
        else:
            return "Healthy"

    df["quadrant"] = df.apply(classify, axis=1)

    action_map = {
        "Reorder Now": "Raise a replenishment order before stock runs out.",
        "Markdown / Clear": "Promote or discount to free up capital.",
        "Watch / Volatile": "Investigate — demand is erratic; review manually.",
        "Healthy": "No action needed; leave as is.",
    }
    df["recommended_action"] = df["quadrant"].map(action_map)

    # --- rupee value at stake ---
    # sales at risk (stockout): units short * price
    units_short = (df["demand_over_lead_time"] - df["projected_position"]).clip(lower=0)
    df["sales_at_risk"] = (units_short * df["unit_price"]).round(2)

    # capital locked (overstock): excess units * unit cost
    excess_units = (df["on_hand_units"].fillna(0) - demand_over_window).clip(lower=0)
    df["capital_locked"] = (excess_units * df["unit_cost"].fillna(df["unit_price"] * 0.5)).round(2)

    df["rupee_value_at_stake"] = np.where(
        df["quadrant"] == "Reorder Now", df["sales_at_risk"],
        np.where(df["quadrant"] == "Markdown / Clear", df["capital_locked"],
                 np.maximum(df["sales_at_risk"], df["capital_locked"]))
    )

    cols = [
        "sku_id", "category", "unit_price", "on_hand_units", "on_order_units",
        "lead_time_days", "reorder_point", "forecast_total_units", "forecast_daily_mean",
        "forecast_lower_total", "forecast_upper_total",
        "stockout_risk", "overstock_risk", "quadrant", "recommended_action",
        "sales_at_risk", "capital_locked", "rupee_value_at_stake",
    ]
    return df[cols].sort_values("rupee_value_at_stake", ascending=False).reset_index(drop=True)


def main():
    forecast, latest_inv, sku_master = load_inputs()
    scored = score_risk(forecast, latest_inv, sku_master)
    scored.to_parquet(PROCESSED / "risk_scored.parquet", index=False)

    print("Risk scoring complete.")
    print(scored["quadrant"].value_counts())
    print(f"\nTotal sales at risk (stockouts): {scored['sales_at_risk'].sum():,.2f}")
    print(f"Total capital locked (overstock): {scored['capital_locked'].sum():,.2f}")
    print(f"\nTop 5 priority SKUs:\n{scored.head(5)[['sku_id','quadrant','rupee_value_at_stake']]}")
    print(f"\nSaved -> {PROCESSED / 'risk_scored.parquet'}")


if __name__ == "__main__":
    main()
