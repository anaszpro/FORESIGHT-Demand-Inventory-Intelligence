# Project FORESIGHT — run instructions

## Setup
```bash
pip install -r requirements.txt
```

## Run the full pipeline (in order — each step depends on the previous)
```bash
python src/generate_data.py   # synthetic dirty extracts -> data/raw/
python src/pipeline.py        # clean + unify + feature-engineer -> data/processed/analysis_ready.parquet
python src/forecast.py        # backtest baseline vs model, train final model, forward forecast
python src/risk.py            # stockout/overstock risk scoring -> data/processed/risk_scored.parquet
```

## Launch the dashboard
```bash
streamlit run app/app.py
```

## Launch the scoring API
```bash
uvicorn service.main:app --reload --port 8000
```
Endpoints: `GET /health`, `GET /skus`, `GET /forecast/{sku_id}`, `POST /forecast/batch`

## Key result files
- `data/processed/data_quality_log.json` — every cleaning decision, logged
- `data/processed/backtest_results.json` — WAPE, model vs seasonal-naive baseline, per fold
- `data/processed/risk_scored.parquet` — final stockout/overstock scores per SKU
