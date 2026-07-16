# Electricity-Demand-Forecasting

Forecasting NSW electricity demand and spot prices 4 steps ahead
on 15-minute interval data, comparing classical time-series models
against deep learning.

## Approach
- **Benchmarks:** naïve, seasonal naïve
- **Classical:** ARIMA/SARIMA, ETS
- **Deep learning:** LSTM & GRU (PyTorch)
- Evaluated on RMSE across train/validation/test splits

## Key findings
- [1-2 lines: which model won, roughly by how much]
- [1 line: the price-spike failure mode — models struggle around
  extreme spikes because X]

## Data
[1 line: source — e.g. AEMO NEM data, 15-min intervals, date range]

## Files
- `NEM Forecasting` — full analysis, all models, charts
