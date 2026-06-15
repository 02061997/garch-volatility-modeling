import numpy as np
import pandas as pd

from volatilitylab.core import forecasts, interval_summary, qlike, regime_slices


def test_qlike_prefers_correct_forecast():
    actual = np.array([0.01, 0.02, 0.015])
    assert qlike(actual, actual) < qlike(actual, actual * 3)


def test_walk_forward_dates_are_out_of_sample():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0, 0.01, 650), index=pd.date_range("2020-01-01", periods=650))
    frame = forecasts(series, window=400, refit_every=50)
    assert frame.index.min() == series.index[401]
    assert frame.forecast_origin.iloc[0] == series.index[400]
    assert len(frame) == len(series) - 401
    assert (frame[["garch", "rolling", "ewma"]] > 0).all().all()


def test_refit_cadence_is_recorded():
    rng = np.random.default_rng(2)
    series = pd.Series(rng.normal(0, 0.01, 520), index=pd.date_range("2019-01-01", periods=520))
    frame = forecasts(series, window=400, refit_every=25)
    refit_offsets = np.flatnonzero(frame.refit.to_numpy())
    assert refit_offsets[0] == 0
    assert set(np.diff(refit_offsets)) == {25}


def test_interval_summary_and_regimes():
    rng = np.random.default_rng(3)
    index = pd.date_range("2019-01-01", periods=1600)
    series = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    frame = forecasts(series, window=400, refit_every=100)
    summary = interval_summary(frame)
    assert 0 <= summary["garch_95_interval_coverage"] <= 1
    regimes = regime_slices(frame)
    assert len(regimes["year_2020"]) > 0
    assert len(regimes["year_2022"]) > 0
    assert len(regimes["normal_ex_2020_2022"]) > 0
