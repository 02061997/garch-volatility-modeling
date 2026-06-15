# Model Card

This repository is an independently reproducible research implementation of
volatility forecasting. It is not investment advice.

## Intended use

- Demonstrate leakage-controlled volatility modeling and diagnostics.
- Compare GARCH(1,1) with EWMA and rolling-volatility baselines.
- Produce reproducible forecast, interval, regime, and parameter-stability
  artifacts.

## Not intended use

- Live risk limits, portfolio margining, or option valuation without additional
  production controls.
- Claims that GARCH dominates all assets, regimes, or objectives.

## Evaluation

The verified run evaluates MAE, RMSE, QLIKE, Diebold-Mariano comparisons,
Ljung-Box tests, residual ARCH effects, AIC/BIC, 95% interval coverage, and
normal/2020/2022 regime slices. Reported results are conditional on the frozen
2015-2023 data snapshots and the committed refit cadence.
