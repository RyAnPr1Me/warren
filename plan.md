# AI Trading CLI — Implementation Plan (Advanced AI Mode + Phased Implementation + GPU Acceleration)

**Goal:** Build a production-grade AI application for **stock prediction** that (1) trains advanced models using **multi-API historical data**, (2) is system-aware and optimized for GPU acceleration, and (3) exposes a robust CLI (`python3 cli.py predict AAPL --15d`, `train`, `backtest`, etc.). Implementation will be executed in clear phases so you can ship iteratively while progressively enabling more advanced modeling and infrastructure.

---

## Data Pipeline

* Integrate **AlphaVantage, FMP, and Finnhub** APIs for 10–20+ years of OHLCV and fundamentals.
* Data fetchers handle rate limits, retries, and incremental updates.
* ETL jobs unify schema, handle splits/dividends, adjust timestamps, and persist in local Parquet + optional cloud store.
* Feature generator produces technical indicators, rolling stats, macro-economic joins, and cross-asset features.

### Data Schema & Storage
| Domain | Table / Object | Key Fields | Notes |
|--------|----------------|-----------|-------|
| Prices | prices_<interval> (Parquet) | symbol, ts | o,h,l,c,adj_close,volume, dividends, splits |
| Fundamentals | fundamentals_snapshot | symbol, period_end | Income, balance, cashflow normalized |
| Earnings | earnings_events | symbol, report_date | Surprise %, EPS actual/estimate |
| Macro | macro_series | series_id, ts | CPI, Fed Funds, VIX, yields |
| Features | feature_matrix_<granularity> | symbol, ts | Engineered features; partitioned by dt |
| Models | registry (MLflow) | model_name, version | Artifacts + metrics |

Local Layout:
```
data/
	raw/<provider>/<symbol>/<year>.json
	staging/<domain>/*.parquet
	warehouse/<table>/dt=YYYY-MM-DD/part-*.parquet
	features/<symbol>/<interval>/dt=YYYY-MM-DD.parquet
	cache/http/<provider>/...
```

Refresh Cadence:
* Daily prices: EOD + optional intraday extension (future).
* Fundamentals: quarterly polling; keep last 8 quarters.
* Macro: daily/weekly depending on series.
* Compaction job monthly to merge small parts.

---

## Phases

### Phase 1 — MVP + GPU enablement (2–3 weeks)

* CLI with `fetch`, `train`, `predict`, `backtest` commands.
* Data ingestion from **multiple APIs** with caching.
* Baseline models: LightGBM + simple LSTM.
* GPU auto-detection + AMP support.
* MLflow tracking.

Deliverables:
* Working CLI (`fetch`, `train`, `predict`, `backtest`).
* Deterministic ingestion (idempotent outputs).
* Baseline metrics: RMSE, MAPE, Directional Accuracy.
* GPU auto-detect + AMP fallback.
* Initial unit tests (>70% data layer).

Acceptance:
* Fetch 5y AAPL <90s cold, <15s warm.
* Baseline train <5m GPU / <12m CPU.
* MLflow run contains params, metrics, artifacts.

### Phase 2 — Advanced Features & Models (3–4 weeks)

* Multi-scale features (daily, weekly, monthly).
* Transformer encoder + TCN architectures.
* Stacking ensembles.
* Feature selection: MI ranking, permutation pruning.

Additions:
* Transformer encoder, TCN stack.
* Ensemble stacker (meta LightGBM/Ridge).
* Drift scan: KS tests vs latest window.
* Feature importance HTML report.

Acceptance:
* +5–10% directional accuracy lift vs Phase 1.
* Feature gen <40% total training wall-clock.
* Reproducible Top-K selection with fixed seed.

### Phase 3 — Training Optimization & Scaling (3–4 weeks)

* PyTorch DDP for multi-GPU.
* Adaptive batch sizing.
* Checkpoint sharding + resumable training.
* K8s job templates.

Additions:
* Resumable checkpoints (<1 epoch loss).
* Elastic batch sizing (utilization heuristic).
* Plateau detector drives LR schedule.

Acceptance:
* 2× speedup vs Phase 2 on same hardware.
* >70% multi-GPU scaling efficiency (2 GPUs).
* Checkpoint size ↓ ≥35% (pre-compression preview).

### Phase 4 — Production & Serving (2–3 weeks)

* Model compression: pruning, quantization, distillation.
* Export: TorchScript, ONNX, TensorRT.
* FastAPI server + monitoring.
* Drift detection & auto-retrain.

Additions:
* FastAPI inference (<40ms p95 GPU / <120ms CPU).
* Canary route + shadow eval.
* Model signature validation.
* Drift dashboard.

Acceptance:
* Zero-downtime deploy verified.
* Live vs backtest divergence <2% 30d.
* Alerting on drift & latency breaches.

---

## GPU & System Optimizations

1. Auto-detect GPU hardware and optimize training parameters.
2. Use mixed-precision (AMP) with `GradScaler`.
3. Gradient accumulation for large batch emulation.
4. Profile training jobs and adapt dynamically.
5. Async checkpoint uploads to avoid I/O stalls.
6. Gradient checkpointing (long seq / Transformer).
7. Torch compile (if stable >10% gain).
8. Dataloader prefetch & pinned memory.
9. Optional FlashAttention (flag).

Profiling Cycle:
* Auto profile every N epochs → hotspot summary → heuristics adjust (batch, workers).

Targets:
* Data loading <15% step time.
* GPU util 70–90% sustained.
* ≥8% VRAM headroom.

Fallback:
* Detect NaNs → disable AMP/compile and mark run.

---

## CLI Configs

* `--gpus auto|0|1|2`
* `--use-amp/--no-amp`
* `--distributed`
* `--profile`
* `--compress-student`

Future Flags:
* `--feature-report`
* `--drift-scan`
* `--resume-run <id>`
* `--export onnx|trt|torchscript`
* `--serve`
* `--backtest-window 365`
* `--target horizon_return|volatility|direction`

Config Precedence: CLI > ENV > config.yaml.

Planned config.yaml skeleton:
```
data:
	root: data
	cache_ttl_minutes: 120
model:
	seq_len: 120
	target: horizon_return
	horizons: [5,10,20]
train:
	batch_size: 256
	epochs: 150
	early_stop_patience: 20
	lr: 5e-4
	optimizer: adamw
feature:
	mi_top_k: 180
	importance_prune_pct: 0.1
serve:
	host: 0.0.0.0
	port: 8080
```

## Feature Engineering Catalog
* Price: log returns, volatility (ATR, Parkinson), z-scores.
* Momentum: RSI, MACD variants, ADX, KAMA.
* Volume/Flow: OBV, VWAP deviation, Chaikin MF.
* Cross-Asset: Sector ETF correlation, VIX beta.
* Macro: Yield curve, CPI delta, risk-on composite.
* Events: Earnings surprise lag, days to event.
* Seasonality: Cyclical encodings.
* Encoding: Target / leave-one-out for regimes.

Quality Gates: null<5%, variance>1e-6, VIF pruning >12 threshold.

## Model Zoo
| Alias | Type | Purpose |
|-------|------|---------|
| lstm_baseline | LSTM | Fast benchmark |
| tcn_stack | TCN | Multi-scale temporal |
| transformer_enc | Transformer | Long-range deps |
| hybrid_fusion | Hybrid | Feature fusion |
| temporal_forest | Tree ensemble | Tabular baseline |
| stacked_ensemble | Meta | Blend predictions |

## Metrics & Evaluation
Regression: RMSE, MAPE, MAE, Spearman IC.
Classification (direction): Accuracy, F1.
Strategy: Sharpe, Max DD, Turnover.
Validation: Purged walk-forward, 5 folds.

Acceptance: Stat sig lift (p<0.05) vs baseline.

## Backtesting
Rules: Long if predicted return >T (+0.5%), optional short < -T.
Costs: Configurable bps, slippage ATR %.
Risk: Vol scaling, stop-loss, max concentration.
Outputs: Equity curve, rolling Sharpe, exposure, turnover.

## MLOps Pipeline
1. Ingestion DAG → validation → feature store.
2. Training trigger (schedule / drift) → Optuna search → register.
3. Staging shadow → A/B → promote.
4. Monitoring: drift, decay, latency, cost.
5. Auto-retrain policy thresholds.

## Observability
Structured JSON logs, Prometheus (GPU/util), OpenTelemetry traces, Grafana dashboards.

## Security
Secrets vault, rate limiting, input validation (^[A-Z.]{1,10}$), dependency scan, SBOM (CycloneDX).

## Testing
Layers: unit, integration, regression, backtest, load.
Coverage targets: 70% → 75% → 80% → 85%+ per phase.

## Structure (Planned)
```
project/
	cli.py
	config/
	data/
	src/
		data/ingestion
		data/validation
		features
		models/architectures
		models/trainers
		evaluation
		serving
		utils
	tests/
		unit
		integration
		backtest
	mlruns/
	notebooks/
```

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Rate limits | Caching, stagger scheduling |
| Leakage | Purged windows, audit checks |
| Overfitting | Early stop, ensembles, regularization |
| GPU instability | Fallback CPU, health probes |
| Cost creep | Budgets, autoscale policies |

## Incremental Strategy
Defer Transformer if Phase 1 slips; single horizon first; stub macro early.

## Success Criteria
Significant predictive lift, stable automated pipeline 30d, production SLO adherence, safe rollback workflows.

---
Expanded plan adds concrete schemas, deliverables, metrics, and operational gates.

---

This plan ensures **data from multiple APIs is always pulled in**, processed, and used in **advanced, GPU-accelerated model training** with a CLI-first workflow.
