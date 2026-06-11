from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def qlike(actual_variance, predicted_variance) -> float:
    a = np.maximum(np.asarray(actual_variance), 1e-12)
    p = np.maximum(np.asarray(predicted_variance), 1e-12)
    return float(np.mean(np.log(p) + a / p))


def forecasts(returns: pd.Series, window: int = 500, step: int = 5) -> pd.DataFrame:
    values = returns.dropna() * 100
    rows = []
    for i in range(window, len(values) - 1, step):
        train = values.iloc[i - window : i]
        model = arch_model(train, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        fit = model.fit(disp="off", show_warning=False)
        garch_var = float(fit.forecast(horizon=1, reindex=False).variance.iloc[0, 0]) / 10_000
        rolling_var = float((train / 100).var(ddof=1))
        ewma_var = float((train / 100).ewm(alpha=0.06, adjust=False).var().iloc[-1])
        actual_var = float((values.iloc[i + 1] / 100) ** 2)
        rows.append(
            {
                "date": values.index[i + 1],
                "actual_variance": actual_var,
                "garch": garch_var,
                "rolling": rolling_var,
                "ewma": ewma_var,
                "omega": float(fit.params["omega"]),
                "alpha": float(fit.params["alpha[1]"]),
                "beta": float(fit.params["beta[1]"]),
                "aic": float(fit.aic),
                "bic": float(fit.bic),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def evaluate(frame: pd.DataFrame) -> dict:
    result = {}
    for model in ["garch", "rolling", "ewma"]:
        error = frame[model] - frame.actual_variance
        result[model] = {
            "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "qlike": qlike(frame.actual_variance, frame[model]),
        }
    return result


def diagnostics(returns: pd.Series) -> dict:
    fit = arch_model(returns.dropna() * 100, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
    residuals = fit.std_resid.dropna()
    lb = acorr_ljungbox(residuals, lags=[10], return_df=True).iloc[0]
    arch = het_arch(residuals, nlags=10)
    return {
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "alpha_plus_beta": float(fit.params["alpha[1]"] + fit.params["beta[1]"]),
        "ljung_box_p": float(lb.lb_pvalue),
        "remaining_arch_p": float(arch[1]),
    }
