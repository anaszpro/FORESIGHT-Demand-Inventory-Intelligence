"""
generate_data.py
Generates 4 synthetic, DELIBERATELY DIRTY extracts for Project FORESIGHT:
  - sku_master.csv
  - calendar.csv
  - sales_daily.csv
  - inventory_snapshots.csv

Dirtiness injected on purpose (mirrors a real client extract):
  - missing values (prices, units, categories)
  - duplicate rows
  - inconsistent category/subcategory labels (casing, whitespace, synonyms)
  - mixed date formats in raw sales export
  - a few negative / absurd outlier values
  - some SKUs with sparse / late-starting history (new launches)

Run: python src/generate_data.py
Output: data/raw/*.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

N_SKUS = 200
START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2025-12-31")  # 2 years of daily history
ALL_DATES = pd.date_range(START_DATE, END_DATE, freq="D")

CATEGORIES = {
    "Furniture": ["Chairs", "Tables", "Sofas", "Shelving"],
    "Decor": ["Wall Art", "Vases", "Candles", "Rugs"],
    "Small Appliances": ["Kettles", "Blenders", "Heaters", "Fans"],
    "Lighting": ["Table Lamps", "Floor Lamps", "String Lights"],
    "Textiles": ["Cushions", "Throws", "Curtains"],
}

# messy label variants injected for a fraction of rows
CATEGORY_DIRTY_VARIANTS = {
    "Furniture": ["furniture", "FURNITURE", " Furniture", "Furnitur"],
    "Decor": ["decor", "Décor", "DECOR ", "Home Decor"],
    "Small Appliances": ["small appliances", "Small Appliance", "SMALL APPLIANCES"],
    "Lighting": ["lighting", "LIGHTING", "Lighting "],
    "Textiles": ["textiles", "Textile", "TEXTILES"],
}


def build_sku_master():
    rows = []
    cats = list(CATEGORIES.keys())
    for i in range(1, N_SKUS + 1):
        sku_id = f"SKU{i:04d}"
        cat = RNG.choice(cats)
        subcat = RNG.choice(CATEGORIES[cat])
        # ~15% of SKUs launched partway through the window (new launches -> sparse history)
        if RNG.random() < 0.15:
            launch_date = START_DATE + pd.Timedelta(days=int(RNG.integers(180, 650)))
        else:
            launch_date = START_DATE - pd.Timedelta(days=int(RNG.integers(0, 400)))
        unit_cost = round(float(RNG.uniform(5, 150)), 2)
        margin_mult = RNG.uniform(1.4, 2.6)
        list_price = round(unit_cost * margin_mult, 2)

        rows.append(
            {
                "sku_id": sku_id,
                "category": cat,
                "subcategory": subcat,
                "launch_date": launch_date,
                "unit_cost": unit_cost,
                "list_price": list_price,
            }
        )
    df = pd.DataFrame(rows)

    # --- inject dirtiness ---
    # 1. inconsistent category labels for a random subset of rows
    dirty_idx = df.sample(frac=0.20, random_state=1).index
    for idx in dirty_idx:
        cat = df.loc[idx, "category"]
        variants = CATEGORY_DIRTY_VARIANTS[cat]
        df.loc[idx, "category"] = RNG.choice(variants)

    # 2. missing unit_cost / list_price for a few SKUs
    miss_idx = df.sample(frac=0.05, random_state=2).index
    df.loc[miss_idx, "unit_cost"] = np.nan

    # 3. a couple of duplicate SKU rows (exact dupes) appended
    dupes = df.sample(n=4, random_state=3)
    df = pd.concat([df, dupes], ignore_index=True)

    # 4. one absurd outlier price (data entry error, e.g. missing decimal)
    outlier_idx = df.sample(n=1, random_state=4).index
    df.loc[outlier_idx, "list_price"] = df.loc[outlier_idx, "list_price"] * 100

    # launch_date as mixed string formats (some ISO, some US-style)
    def messy_date(d):
        if RNG.random() < 0.3:
            return d.strftime("%m/%d/%Y")
        return d.strftime("%Y-%m-%d")

    df["launch_date"] = df["launch_date"].apply(messy_date)

    return df


def build_calendar():
    df = pd.DataFrame({"date": ALL_DATES})
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(
        {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall",
        }
    )
    # simple holiday calendar (a handful of fixed dates + Black Friday-ish)
    holidays = set()
    for yr in range(START_DATE.year, END_DATE.year + 1):
        holidays.update(
            [
                pd.Timestamp(yr, 1, 1),
                pd.Timestamp(yr, 12, 25),
                pd.Timestamp(yr, 11, 28),  # approx Black Friday
                pd.Timestamp(yr, 7, 4),
            ]
        )
    df["is_holiday"] = df["date"].isin(holidays).astype(int)

    # promo events: ~6 promo windows per year, each 4-7 days
    promo_event = pd.Series([""] * len(df), index=df.index)
    n_years = END_DATE.year - START_DATE.year + 1
    for yr in range(START_DATE.year, END_DATE.year + 1):
        for name, month, day in [
            ("New Year Sale", 1, 5),
            ("Spring Refresh", 4, 10),
            ("Summer Clearance", 7, 15),
            ("Back to Home", 9, 5),
            ("Black Friday", 11, 25),
            ("Holiday Sale", 12, 15),
        ]:
            try:
                start = pd.Timestamp(yr, month, day)
            except ValueError:
                continue
            length = int(RNG.integers(4, 8))
            mask = (df["date"] >= start) & (df["date"] < start + pd.Timedelta(days=length))
            promo_event.loc[mask] = name
    df["promo_event"] = promo_event
    return df


def build_sales_and_inventory(sku_master, calendar):
    # use clean version of sku_master internally for generation logic
    clean_sku = sku_master.drop_duplicates(subset="sku_id").copy()
    clean_sku["category_clean"] = clean_sku["category"].str.strip().str.title()
    clean_sku["launch_date_parsed"] = pd.to_datetime(
        clean_sku["launch_date"], format="mixed", errors="coerce"
    )

    sales_rows = []
    inv_rows = []

    cal = calendar.set_index("date")

    for _, sku in clean_sku.iterrows():
        sku_id = sku["sku_id"]
        launch = sku["launch_date_parsed"]
        base_demand = RNG.uniform(2, 40)  # avg units/day baseline
        trend = RNG.uniform(-0.0005, 0.0008)  # slow drift
        promo_lift = RNG.uniform(1.3, 2.2)
        weekend_lift = RNG.uniform(1.05, 1.4)
        noise_sigma = base_demand * RNG.uniform(0.25, 0.5)
        # seasonality: some categories are seasonal (e.g. Heaters/Fans), others flat
        seasonal_amp = base_demand * RNG.uniform(0.1, 0.6)
        seasonal_phase = RNG.uniform(0, 2 * np.pi)

        price = sku["list_price"] if pd.notna(sku["list_price"]) else RNG.uniform(20, 200)
        cost = sku["unit_cost"] if pd.notna(sku["unit_cost"]) else price * 0.5

        # on-hand stock simulation state
        on_hand = int(RNG.integers(20, 200))
        lead_time = int(RNG.integers(7, 30))
        # under-target reorder point (vs actual mean demand) for a subset of SKUs
        # so some genuinely run tight on stock -> realistic stockout risk
        rp_mult = RNG.choice([0.6, 0.9, 1.2, 1.5], p=[0.2, 0.3, 0.3, 0.2])
        reorder_point = int(base_demand * lead_time * rp_mult)
        reorder_qty_mult = RNG.uniform(1.2, 2.0)
        pending_orders = []  # list of (arrival_date, qty) — orders in transit

        day_idx = 0
        for date in ALL_DATES:
            if date < launch:
                continue
            t = day_idx
            day_idx += 1

            # receive any orders that have arrived
            arrived_qty = sum(q for d, q in pending_orders if d <= date)
            if arrived_qty:
                on_hand += arrived_qty
                pending_orders = [(d, q) for d, q in pending_orders if d > date]

            dow = date.dayofweek
            is_weekend = dow >= 5
            promo = cal.loc[date, "promo_event"] != ""
            is_holiday = cal.loc[date, "is_holiday"] == 1

            seasonal = seasonal_amp * np.sin(2 * np.pi * (t / 365.0) + seasonal_phase)
            mean_demand = base_demand + trend * t + seasonal
            if is_weekend:
                mean_demand *= weekend_lift
            if promo:
                mean_demand *= promo_lift
            if is_holiday:
                mean_demand *= 1.15
            mean_demand = max(mean_demand, 0.5)

            units = RNG.poisson(mean_demand)
            # stock-out suppression: can't sell more than on_hand
            units_sold = min(units, on_hand)

            revenue = round(units_sold * price, 2)
            sales_rows.append(
                {
                    "date": date,
                    "sku_id": sku_id,
                    "units_sold": units_sold,
                    "revenue": revenue,
                    "unit_price": price,
                    "promo_flag": int(promo),
                }
            )

            on_hand -= units_sold
            on_hand = max(on_hand, 0)

            # place a reorder if below reorder point and nothing already in transit
            already_ordering = len(pending_orders) > 0
            if on_hand <= reorder_point and not already_ordering:
                qty = int(mean_demand * lead_time * reorder_qty_mult)
                arrival = date + pd.Timedelta(days=lead_time)
                pending_orders.append((arrival, qty))

            on_order_total = sum(q for d, q in pending_orders)

            # weekly inventory snapshot (not daily, to mirror periodic counts)
            if date.dayofweek == 0:  # Monday snapshot
                inv_rows.append(
                    {
                        "date": date,
                        "sku_id": sku_id,
                        "on_hand_units": max(on_hand, 0),
                        "on_order_units": on_order_total,
                        "lead_time_days": lead_time,
                        "reorder_point": reorder_point,
                    }
                )

    sales_df = pd.DataFrame(sales_rows)
    inv_df = pd.DataFrame(inv_rows)

    # --- inject dirtiness into sales_daily ---
    # 1. duplicate ~1% of rows
    dupes = sales_df.sample(frac=0.01, random_state=5)
    sales_df = pd.concat([sales_df, dupes], ignore_index=True)

    # 2. missing unit_price for some rows
    miss_idx = sales_df.sample(frac=0.03, random_state=6).index
    sales_df.loc[miss_idx, "unit_price"] = np.nan

    # 3. a few negative units_sold (data entry / return errors)
    neg_idx = sales_df.sample(n=25, random_state=7).index
    sales_df.loc[neg_idx, "units_sold"] = -sales_df.loc[neg_idx, "units_sold"].abs()

    # 4. a few extreme outlier units_sold (fat-finger entry)
    out_idx = sales_df.sample(n=10, random_state=8).index
    sales_df.loc[out_idx, "units_sold"] = sales_df.loc[out_idx, "units_sold"] * 50 + 500

    # 5. mixed date formats (export inconsistency) - convert date col to messy strings
    def messy_date(d):
        if RNG.random() < 0.15:
            return pd.Timestamp(d).strftime("%d-%m-%Y")
        elif RNG.random() < 0.15:
            return pd.Timestamp(d).strftime("%m/%d/%Y")
        return pd.Timestamp(d).strftime("%Y-%m-%d")

    sales_df["date"] = sales_df["date"].apply(messy_date)

    # --- inject dirtiness into inventory_snapshots ---
    miss_idx2 = inv_df.sample(frac=0.02, random_state=9).index
    inv_df.loc[miss_idx2, "on_order_units"] = np.nan
    dupes2 = inv_df.sample(frac=0.01, random_state=10)
    inv_df = pd.concat([inv_df, dupes2], ignore_index=True)
    inv_df["date"] = pd.to_datetime(inv_df["date"]).dt.strftime("%Y-%m-%d")

    return sales_df, inv_df


def main():
    print("Generating sku_master...")
    sku_master = build_sku_master()
    print("Generating calendar...")
    calendar = build_calendar()
    print("Generating sales_daily and inventory_snapshots (this simulates ~200 SKUs x 2 years)...")
    sales_daily, inventory_snapshots = build_sales_and_inventory(sku_master, calendar)

    sku_master.to_csv(OUT / "sku_master.csv", index=False)
    calendar.assign(date=calendar["date"].dt.strftime("%Y-%m-%d")).to_csv(
        OUT / "calendar.csv", index=False
    )
    sales_daily.to_csv(OUT / "sales_daily.csv", index=False)
    inventory_snapshots.to_csv(OUT / "inventory_snapshots.csv", index=False)

    print("\nDone. Files written to", OUT)
    print("sku_master:", sku_master.shape)
    print("calendar:", calendar.shape)
    print("sales_daily:", sales_daily.shape)
    print("inventory_snapshots:", inventory_snapshots.shape)


if __name__ == "__main__":
    main()
