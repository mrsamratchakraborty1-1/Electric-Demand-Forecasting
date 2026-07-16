# Electricity Demand & Price Forecasting — Australian NEM (NSW)

Forecasting NSW electricity **demand (MW)** and **wholesale spot price ($/MWh)**
4 steps ahead (one hour) on 15-minute AEMO interval data — benchmarking
classical time-series models against deep learning, with a strict
leakage-aware evaluation design.

**Headline result:** demand and price need fundamentally different models.
Stable, seasonal demand is won by classical Holt-Winters; spike-driven,
event-based price is won by a recurrent net (GRU).

## Approach
- **Baselines:** naïve, seasonal naïve (M = 96 intraday cycle)
- **Classical:** Ridge (direct multi-step, 7 AR lags + cyclical calendar
  features), SES, Holt-Winters additive (daily & weekly seasonality)
- **Deep learning (PyTorch):** MLP, 2-layer stacked LSTM, GRU — all direct
  multi-step (no recursive error accumulation), grid-searched over
  lookback/width/dropout with early stopping & gradient clipping

## Results (test RMSE)
| Target | Best model | Test RMSE |
|---|---|---|
| Demand (MW) | **Holt-Winters, M = 96** | **255.9 MW** |
| Spot price ($/MWh) | **GRU** | **454.2 $/MWh** |

- Demand: Holt-Winters beat every neural model — strong stable intraday
  seasonality favours structured seasonal modelling over gated networks
  on a short (5-week) training window.
- Price: linear/level models (Ridge, SES) looked great in validation and
  collapsed on test spikes (~$11,000/MWh events); GRU's gating handled
  regime shifts best, though spike magnitudes remain under-forecast.

## Design details worth noting
- **Leakage prevention:** chronological 70/15/15 split *before* cleaning;
  scalers fit on train only; rolling non-overlapping 4-step evaluation
- **Negative prices (23.7% of observations):** handled via a signed-log
  transform, sign(x)·log(1+|x|); RMSE computed back in original units
- **Outlier repair:** seasonal-naïve imputation (lag-96) on training data only
- **Honest failure analysis:** no model anticipates exogenous market shocks
  from price + calendar features alone — documented, not hidden

## Data
AEMO NEM data, NSW region, 15-minute settlement intervals
(TOTALDEMAND, RRP), Jan–Feb 2026. ~3,800 observations.

## Files
- `NEM Forecasting` — full pipeline: EDA, models, evaluation

*QBUS6840 group project (University of Sydney). My contributions:
Coding for Ridge Regression and LSTM Model, writing on the relevant part of the report, including model description and results
