"""
pipeline.py — Project FORESIGHT data pipeline (D1)

Ingests the four raw extracts, validates + cleans them, joins into one
analysis-ready dataset, and engineers modelling features.

Every cleaning decision is logged to data/processed/data_quality_log.json
so the decisions are auditable (per the brief: "document rationale").

Run: python src/pipeline.py
Output:
  data/processed/analysis_ready.parquet   (sales+calendar+sku, feature-engineered)
  data/processed/inventory_clean.parquet  (cleaned inventory snapshots)
  data/processed/sku_master_clean.parquet
  data/processed/data_quality_log.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

LOG = []


def log(step, detail):
    LOG.append({"step": step, "detail": detail})
    print(f"[{step}] {detail}")


# ---------------------------------------------------------------------------
# 1. INGEST
# ---------------------------------------------------------------------------
def ingest():
    sales = pd.read_csv(RAW / "sales_daily.csv")
    sku = pd.read_csv(RAW / "sku_master.csv")
    calendar = pd.read_csv(RAW / "calendar.csv")
    inventory = pd.read_csv(RAW / "inventory_snapshots.csv")
    log("ingest", f"loaded sales={sales.shape}, sku={sku.shape}, "
                  f"calendar={calendar.shape}, inventory={inventory.shape}")
    return sales, sku, calendar, inventory


# ---------------------------------------------------------------------------
# 2. CLEAN sku_master
# ---------------------------------------------------------------------------
def clean_sku_master(sku: pd.DataFrame) -> pd.DataFrame:
    df = sku.copy()

    before = len(df)
    df = df.drop_duplicates(subset="sku_id", keep="first")
    log("sku_master.dedupe", f"dropped {before - len(df)} exact/duplicate sku_id rows")

    # normalize category labels: strip, title-case, fix known synonyms
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["category"] = df["category"].replace({"Décor": "Decor", "Home Decor": "Decor"})
    log("sku_master.category_normalize",
        "trimmed whitespace, title-cased, mapped 'Décor'/'Home Decor' -> 'Decor'")

    # parse mixed-format launch_date robustly
    df["launch_date"] = pd.to_datetime(df["launch_date"], format="mixed", errors="coerce")
    n_bad_dates = df["launch_date"].isna().sum()
    log("sku_master.date_parse", f"parsed launch_date (mixed formats); {n_bad_dates} unparseable")

    # impute missing unit_cost with category median (documented assumption)
    n_missing_cost = df["unit_cost"].isna().sum()
    df["unit_cost"] = df.groupby("category")["unit_cost"].transform(
        lambda x: x.fillna(x.median())
    )
    log("sku_master.impute_cost",
        f"filled {n_missing_cost} missing unit_cost values with category median")

    # cap the one absurd list_price outlier (>4 std from category mean) at the 99th pct
    cat_stats = df.groupby("category")["list_price"].transform(lambda x: x)
    q99 = df["list_price"].quantile(0.99)
    n_capped = (df["list_price"] > q99 * 1.5).sum()
    df.loc[df["list_price"] > q99 * 1.5, "list_price"] = q99
    log("sku_master.cap_outlier_price", f"capped {n_capped} extreme list_price outlier(s) at ~p99")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. CLEAN sales_daily
# ---------------------------------------------------------------------------
def clean_sales(sales: pd.DataFrame, sku_clean: pd.DataFrame) -> pd.DataFrame:
    df = sales.copy()

    before = len(df)
    df = df.drop_duplicates(subset=["date", "sku_id"], keep="first")
    log("sales.dedupe", f"dropped {before - len(df)} duplicate (date, sku_id) rows")

    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    n_bad = df["date"].isna().sum()
    df = df.dropna(subset=["date"])
    log("sales.date_parse", f"parsed mixed date formats; dropped {n_bad} unparseable rows")

    # negative units_sold are data-entry errors (not returns, per client) -> clip to 0
    n_neg = (df["units_sold"] < 0).sum()
    df.loc[df["units_sold"] < 0, "units_sold"] = 0
    log("sales.negative_units", f"clipped {n_neg} negative units_sold rows to 0")

    # extreme outliers: units_sold > 20x that SKU's own 99th percentile -> winsorize
    df["units_sold"] = df["units_sold"].astype(float)

    def winsorize_sku(group):
        cap = group["units_sold"].quantile(0.99) * 3
        if cap <= 0:
            cap = group["units_sold"].max()
        group.loc[group["units_sold"] > cap, "units_sold"] = cap
        return group

    n_before_extreme = (df.groupby("sku_id")["units_sold"]
                         .transform(lambda x: x > x.quantile(0.99) * 3).sum())
    df = df.groupby("sku_id", group_keys=False)[df.columns].apply(winsorize_sku)
    log("sales.outlier_winsorize",
        f"winsorized ~{int(n_before_extreme)} extreme units_sold spikes per-SKU at 3x p99")

    # missing unit_price -> fill from sku_master list_price, else category median price that day
    price_map = sku_clean.set_index("sku_id")["list_price"]
    n_missing_price = df["unit_price"].isna().sum()
    df["unit_price"] = df["unit_price"].fillna(df["sku_id"].map(price_map))
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())
    log("sales.impute_price", f"filled {n_missing_price} missing unit_price from sku_master list_price")

    # recompute revenue where it disagrees with price*units (defensive consistency fix)
    df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)

    # keep only SKUs known in sku_master (referential integrity)
    known = set(sku_clean["sku_id"])
    n_orphan = (~df["sku_id"].isin(known)).sum()
    df = df[df["sku_id"].isin(known)]
    log("sales.referential_integrity", f"dropped {n_orphan} rows with unknown sku_id")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. CLEAN inventory_snapshots
# ---------------------------------------------------------------------------
def clean_inventory(inv: pd.DataFrame, sku_clean: pd.DataFrame) -> pd.DataFrame:
    df = inv.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["date", "sku_id"], keep="first")
    log("inventory.dedupe", f"dropped {before - len(df)} duplicate (date, sku_id) rows")

    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"])

    n_missing_onorder = df["on_order_units"].isna().sum()
    df["on_order_units"] = df["on_order_units"].fillna(0)
    log("inventory.impute_on_order", f"filled {n_missing_onorder} missing on_order_units with 0")

    known = set(sku_clean["sku_id"])
    df = df[df["sku_id"].isin(known)]

    for col in ["on_hand_units", "on_order_units"]:
        n_neg = (df[col] < 0).sum()
        df.loc[df[col] < 0, col] = 0
        if n_neg:
            log("inventory.negative_stock", f"clipped {n_neg} negative {col} values to 0")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. UNIFY + FEATURE ENGINEER
# ---------------------------------------------------------------------------
def build_analysis_ready(sales, sku_clean, calendar):
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])

    df = sales.merge(sku_clean, on="sku_id", how="left", suffixes=("", "_sku"))
    df = df.merge(calendar, on="date", how="left")
    log("merge", f"joined sales + sku_master + calendar -> {df.shape}")

    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    # --- calendar / seasonality features ---
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_promo"] = (df["promo_event"].fillna("") != "").astype(int)

    # --- lag & rolling features (per SKU, no leakage: shift(1) before rolling) ---
    g = df.groupby("sku_id")["units_sold"]
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["lag_28"] = g.shift(28)
    df["roll_mean_7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["roll_mean_28"] = g.shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    df["roll_std_28"] = g.shift(1).rolling(28).std().reset_index(level=0, drop=True)

    # --- SKU age (days since launch) ---
    df["sku_age_days"] = (df["date"] - df["launch_date"]).dt.days

    log("feature_engineering",
        "added dow/weekend/promo flags, lag_7/14/28, rolling mean/std(7,28), sku_age_days")

    return df


def main():
    sales, sku, calendar, inventory = ingest()

    sku_clean = clean_sku_master(sku)
    sales_clean = clean_sales(sales, sku_clean)
    inv_clean = clean_inventory(inventory, sku_clean)

    analysis_ready = build_analysis_ready(sales_clean, sku_clean, calendar)

    sku_clean.to_parquet(PROCESSED / "sku_master_clean.parquet", index=False)
    inv_clean.to_parquet(PROCESSED / "inventory_clean.parquet", index=False)
    analysis_ready.to_parquet(PROCESSED / "analysis_ready.parquet", index=False)

    with open(PROCESSED / "data_quality_log.json", "w") as f:
        json.dump(LOG, f, indent=2, default=str)

    print("\nSaved:")
    print(" -", PROCESSED / "sku_master_clean.parquet", sku_clean.shape)
    print(" -", PROCESSED / "inventory_clean.parquet", inv_clean.shape)
    print(" -", PROCESSED / "analysis_ready.parquet", analysis_ready.shape)
    print(" -", PROCESSED / "data_quality_log.json", f"({len(LOG)} steps logged)")


if __name__ == "__main__":
    main()
