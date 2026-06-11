# GARCH Volatility Modeling

Walk-forward comparison of GARCH(1,1), EWMA, and rolling-variance forecasts
with residual diagnostics and QLIKE evaluation. Full results use frozen
2015–2023 market-data snapshots for SPY and five equities.

```bash
uv sync
make test
make reproduce-results
```
