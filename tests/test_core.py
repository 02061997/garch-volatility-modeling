import numpy as np
import pandas as pd

from volatilitylab.core import forecasts, qlike


def test_qlike_prefers_correct_forecast():
    actual = np.array([0.01, 0.02, 0.015])
    assert qlike(actual, actual) < qlike(actual, actual * 3)


def test_walk_forward_dates_are_out_of_sample():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0, 0.01, 650), index=pd.date_range("2020-01-01", periods=650))
    frame = forecasts(series, window=400, step=50)
    assert frame.index.min() == series.index[401]
    assert (frame[["garch", "rolling", "ewma"]] > 0).all().all()
