# Verified Results

## Reproduction status

`make reproduce-results` completed locally on June 15, 2026 using frozen Yahoo
Finance snapshots. The run produced 10,572 daily one-step-ahead forecast rows
across six symbols. GARCH parameters are refit every 21 forecast origins and
daily forecasts update recursively from returns available at the forecast
origin.

| Ticker | Best RMSE model | RMSE | Best QLIKE model | QLIKE | GARCH 95% coverage |
|---|---|---:|---|---:|---:|
| AAPL | garch | 0.000915 | garch | -7.1308 | 94.3% |
| JNJ | ewma | 0.000482 | garch | -7.9551 | 93.9% |
| JPM | garch | 0.001193 | garch | -7.3974 | 94.2% |
| MSFT | garch | 0.000885 | garch | -7.2766 | 93.6% |
| SPY | garch | 0.000470 | garch | -8.3974 | 94.6% |
| XOM | ewma | 0.000973 | ewma | -7.2580 | 93.9% |

GARCH now has the best RMSE for AAPL, JPM, MSFT, and SPY and the best QLIKE for
five of six symbols. EWMA remains best for JNJ and XOM RMSE and XOM QLIKE. The
result supports GARCH as a strong default in this sample, not a universal
winner.

`reports/latest/statistical_tests.json` contains residual diagnostics,
Diebold-Mariano comparisons against EWMA and rolling volatility, regime slices
for normal periods, 2020, and 2022, and refit counts. The committed figures
show forecast volatility and parameter stability.
