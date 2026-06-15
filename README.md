# GARCH Volatility Modeling

Walk-forward comparison of GARCH(1,1), EWMA, and rolling-variance forecasts
with residual diagnostics and QLIKE evaluation. Full results use frozen
2015–2023 market-data snapshots for SPY and five equities.

## What is implemented

- Daily one-step-ahead volatility forecasts with no future returns in the
  forecast origin.
- GARCH(1,1), EWMA, and rolling-variance baselines.
- Refit cadence tracking, parameter-stability evidence, and 95% forecast
  interval coverage.
- Normal-period, 2020, and 2022 regime slices.
- AIC, BIC, Ljung-Box, residual ARCH diagnostics, QLIKE, RMSE, MAE, and
  Diebold-Mariano comparisons.

```bash
uv sync
make test
make reproduce-smoke
make reproduce-results
```
