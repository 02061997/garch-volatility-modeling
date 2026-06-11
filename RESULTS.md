# Verified Results

## Reproduction status

`make reproduce-results` completed locally on June 11, 2026 using frozen Yahoo
Finance snapshots. Forecasts are strictly one-step-ahead and refitted without
using future observations.

| Ticker | Best RMSE model | RMSE | Best QLIKE model | QLIKE |
|---|---|---:|---|---:|
| AAPL | rolling | 0.000688 | rolling | -6.7270 |
| JNJ | garch | 0.000380 | ewma | -7.9921 |
| JPM | garch | 0.000328 | garch | -7.7149 |
| MSFT | rolling | 0.000454 | rolling | -7.3425 |
| SPY | garch | 0.000193 | garch | -8.5354 |
| XOM | ewma | 0.000680 | garch | -6.9239 |

No model dominates every asset and metric. GARCH has the lowest RMSE for JNJ,
JPM, SPY, and XOM; rolling volatility is best for AAPL and MSFT. QLIKE rankings
also vary. This is the principal negative result and should guide model
selection rather than a blanket claim that GARCH is superior.
