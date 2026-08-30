"""
service/main.py — Project FORESIGHT scoring service (D6)

Serves forecast + risk for a given SKU (or a batch of SKUs).
Reads pre-computed forecast/risk tables (fast, no live model inference needed
for this scope) and falls back gracefully on bad input.

Run locally:
  uvicorn service.main:app --reload --port 8000

Then:
  GET  /health
  GET  /forecast/{sku_id}
  POST /forecast/batch    body: {"sku_ids": ["SKU0001", "SKU0002"]}
  GET  /skus              list all known SKU ids
"""

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

app = FastAPI(
    title="FORESIGHT Scoring Service",
    description="Forecast + stockout/overstock risk for NorthBay Living SKUs.",
    version="1.0",
)

_risk_df = None


def get_risk_df():
    global _risk_df
    if _risk_df is None:
        path = PROCESSED / "risk_scored.parquet"
        if not path.exists():
            raise FileNotFoundError(
                "risk_scored.parquet not found — run the pipeline first "
                "(pipeline.py -> forecast.py -> risk.py)."
            )
        _risk_df = pd.read_parquet(path).set_index("sku_id")
    return _risk_df


class BatchRequest(BaseModel):
    sku_ids: List[str]


def row_to_dict(sku_id: str, row: pd.Series) -> dict:
    return {
        "sku_id": sku_id,
        "category": row["category"],
        "forecast_total_units_next_horizon": float(row["forecast_total_units"]),
        "forecast_daily_mean": float(row["forecast_daily_mean"]),
        "forecast_interval": {
            "lower": float(row["forecast_lower_total"]),
            "upper": float(row["forecast_upper_total"]),
        },
        "inventory": {
            "on_hand_units": float(row["on_hand_units"]),
            "on_order_units": float(row["on_order_units"]),
            "lead_time_days": float(row["lead_time_days"]),
        },
        "risk": {
            "stockout_risk": round(float(row["stockout_risk"]), 3),
            "overstock_risk": round(float(row["overstock_risk"]), 3),
            "quadrant": row["quadrant"],
            "recommended_action": row["recommended_action"],
        },
        "rupee_impact": {
            "sales_at_risk": float(row["sales_at_risk"]),
            "capital_locked": float(row["capital_locked"]),
            "value_at_stake": float(row["rupee_value_at_stake"]),
        },
    }


@app.get("/health")
def health():
    try:
        df = get_risk_df()
        return {"status": "ok", "skus_loaded": int(len(df))}
    except FileNotFoundError as e:
        return {"status": "degraded", "detail": str(e)}


@app.get("/skus")
def list_skus():
    try:
        df = get_risk_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"count": len(df), "sku_ids": df.index.tolist()}


@app.get("/forecast/{sku_id}")
def get_forecast(sku_id: str):
    try:
        df = get_risk_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    sku_id_norm = sku_id.strip().upper()
    if sku_id_norm not in df.index:
        raise HTTPException(status_code=404, detail=f"Unknown sku_id '{sku_id}'.")

    return row_to_dict(sku_id_norm, df.loc[sku_id_norm])


@app.post("/forecast/batch")
def get_forecast_batch(req: BatchRequest):
    try:
        df = get_risk_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not req.sku_ids:
        raise HTTPException(status_code=400, detail="sku_ids must be a non-empty list.")

    results, not_found = [], []
    for sku_id in req.sku_ids:
        sku_id_norm = str(sku_id).strip().upper()
        if sku_id_norm in df.index:
            results.append(row_to_dict(sku_id_norm, df.loc[sku_id_norm]))
        else:
            not_found.append(sku_id)

    return {"results": results, "not_found": not_found}
