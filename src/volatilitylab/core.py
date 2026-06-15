from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats


def qlike(actual_variance, predicted_variance) -> float:
    a = np.maximum(np.asarray(actual_variance), 1e-12)
    p = np.maximum(np.asarray(predicted_variance), 1e-12)
    return float(np.mean(np.log(p) + a / p))


def forecasts(returns: pd.Series, window: int = 500, refit_every: int = 21) -> pd.DataFrame:
    values = returns.dropna() * 100
    decimal = values / 100
    rows = []
    fit = None
    variance_pct = None
    for i in range(window, len(values) - 1):
        refit = fit is None or (i - window) % refit_every == 0
        if refit:
            train = values.iloc[i - window : i]
            model = arch_model(train, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
            fit = model.fit(disp="off", show_warning=False)
            variance_pct = float(fit.conditional_volatility.iloc[-1] ** 2)
        assert fit is not None and variance_pct is not None
        omega = float(fit.params["omega"])
        alpha = float(fit.params["alpha[1]"])
        beta = float(fit.params["beta[1]"])
        variance_pct = omega + alpha * float(values.iloc[i] ** 2) + beta * variance_pct
        garch_var = variance_pct / 10_000
        train_decimal = decimal.iloc[i - window : i]
        rolling_var = float(train_decimal.var(ddof=1))
        ewma_var = float(train_decimal.ewm(alpha=0.06, adjust=False).var().iloc[-1])
        actual_return = float(decimal.iloc[i + 1])
        actual_var = actual_return**2
        forecast_std = float(np.sqrt(garch_var))
        lower, upper = -1.96 * forecast_std, 1.96 * forecast_std
        rows.append(
            {
                "date": values.index[i + 1],
                "forecast_origin": values.index[i],
                "actual_return": actual_return,
                "actual_variance": actual_var,
                "garch": garch_var,
                "rolling": rolling_var,
                "ewma": ewma_var,
                "garch_lower_95": lower,
                "garch_upper_95": upper,
                "garch_interval_covered": lower <= actual_return <= upper,
                "omega": omega,
                "alpha": alpha,
                "beta": beta,
                "alpha_plus_beta": alpha + beta,
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "refit": refit,
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


def diebold_mariano(
    actual_variance: pd.Series, model_a: pd.Series, model_b: pd.Series
) -> dict[str, float]:
    loss_diff = (model_a - actual_variance) ** 2 - (model_b - actual_variance) ** 2
    loss_diff = loss_diff.dropna()
    std = loss_diff.std(ddof=1)
    statistic = loss_diff.mean() / (std / np.sqrt(len(loss_diff))) if std > 0 else 0.0
    pvalue = 2 * (1 - stats.t.cdf(abs(statistic), df=len(loss_diff) - 1))
    return {"statistic": float(statistic), "pvalue": float(pvalue)}


def regime_slices(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(frame.index)
    return {
        "normal_ex_2020_2022": frame[(index.year != 2020) & (index.year != 2022)],
        "year_2020": frame[index.year == 2020],
        "year_2022": frame[index.year == 2022],
    }


def regime_metrics(frame: pd.DataFrame) -> dict[str, dict]:
    return {name: evaluate(group) for name, group in regime_slices(frame).items() if len(group) > 0}


def interval_summary(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "garch_95_interval_coverage": float(frame.garch_interval_covered.mean()),
        "mean_garch_annualized_volatility": float(np.sqrt(frame.garch).mean() * np.sqrt(252)),
        "mean_realized_proxy_annualized_volatility": float(
            np.sqrt(frame.actual_variance).mean() * np.sqrt(252)
        ),
    }
