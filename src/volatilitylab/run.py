from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from .artifacts import environment, run_dir, save_json, sha256
from .core import diebold_mariano, diagnostics, evaluate, forecasts, interval_summary, regime_metrics


SYMBOLS = ["SPY", "AAPL", "MSFT", "JPM", "XOM", "JNJ"]


def publish_latest(out: Path) -> None:
    latest = Path("reports/latest")
    latest.mkdir(parents=True, exist_ok=True)
    for existing in latest.iterdir():
        if existing.is_file():
            existing.unlink()
    for source in out.iterdir():
        if source.is_file():
            shutil.copy2(source, latest / source.name)
    Path("reports/SOURCE_RUN.txt").write_text(f"{out}\n")


def synthetic(n=900):
    rng = np.random.default_rng(7)
    var = np.empty(n)
    ret = np.empty(n)
    var[0] = 0.0001
    ret[0] = rng.normal(0, np.sqrt(var[0]))
    for i in range(1, n):
        var[i] = 0.000002 + 0.08 * ret[i - 1] ** 2 + 0.9 * var[i - 1]
        ret[i] = rng.normal(0, np.sqrt(var[i]))
    return pd.Series(ret, index=pd.date_range("2018-01-01", periods=n))


def load(symbol):
    path = Path("data") / f"{symbol}_2015_2023.csv"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        raw = yf.download(
            symbol, start="2015-01-01", end="2024-01-01", auto_adjust=True, progress=False
        )
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.rename("Close").to_frame().to_csv(path, index_label="Date")
    price = pd.read_csv(path, parse_dates=["Date"], index_col="Date").iloc[:, 0]
    return np.log(price).diff().dropna(), {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(price),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = run_dir("garch-volatility-modeling", args.smoke)
    all_predictions, metrics, tests, manifests = [], {}, {}, []
    symbols = ["SYNTH"] if args.smoke else SYMBOLS
    for symbol in symbols:
        series, manifest = (synthetic(), {"source": "synthetic"}) if args.smoke else load(symbol)
        window = 400 if args.smoke else 500
        refit_every = 20 if args.smoke else 21
        frame = forecasts(series, window=window, refit_every=refit_every)
        metrics[symbol] = evaluate(frame)
        metrics[symbol]["intervals"] = interval_summary(frame)
        tests[symbol] = {
            "diagnostics": diagnostics(series),
            "die_bold_mariano_garch_vs_rolling": diebold_mariano(
                frame["actual_variance"], frame["garch"], frame["rolling"]
            ),
            "die_bold_mariano_garch_vs_ewma": diebold_mariano(
                frame["actual_variance"], frame["garch"], frame["ewma"]
            ),
            "regime_metrics": regime_metrics(frame),
            "refit_count": int(frame.refit.sum()),
            "forecast_rows": int(len(frame)),
        }
        manifests.append({"symbol": symbol, **manifest})
        all_predictions.append(frame.assign(symbol=symbol).reset_index())
    pred = pd.concat(all_predictions)
    pred.to_parquet(out / "predictions.parquet", index=False)
    spy = pred[pred.symbol == symbols[0]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        spy.date, np.sqrt(spy.actual_variance) * np.sqrt(252), alpha=0.5, label="realized proxy"
    )
    ax.plot(spy.date, np.sqrt(spy.garch) * np.sqrt(252), label="GARCH")
    ax.legend()
    ax.set_ylabel("Annualized volatility")
    fig.tight_layout()
    fig.savefig(out / "volatility_forecast.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for symbol, group in pred.groupby("symbol"):
        ax.plot(group.date, group.alpha_plus_beta, label=symbol, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("GARCH alpha + beta")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "parameter_stability.png", dpi=180)
    plt.close(fig)

    save_json(out / "config.yaml", {"window": 500, "refit_every_days": 21})
    save_json(out / "environment.json", environment())
    save_json(out / "data_manifest.json", manifests)
    save_json(out / "metrics.json", metrics)
    save_json(out / "statistical_tests.json", tests)
    (out / "run.log").write_text("completed\n")
    if not args.smoke:
        publish_latest(out)
    print(out)


if __name__ == "__main__":
    main()
