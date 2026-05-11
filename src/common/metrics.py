"""Reusable error metrics for model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


ArrayLike = pd.Series | np.ndarray | list[float]


def as_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def rmse(error: ArrayLike) -> float:
    arr = as_float_array(error)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr**2)))


def mae(error: ArrayLike) -> float:
    arr = as_float_array(error)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    actual_arr = as_float_array(actual)
    pred_arr = as_float_array(predicted)
    valid = np.abs(actual_arr) > 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred_arr[valid] - actual_arr[valid]) / np.abs(actual_arr[valid])) * 100)


def direction_hit_rate(actual: pd.Series, predicted: pd.Series) -> float:
    actual_diff = actual.diff().iloc[1:]
    pred_diff = predicted.diff().iloc[1:]
    if actual_diff.empty:
        return float("nan")
    return float((np.sign(actual_diff) == np.sign(pred_diff)).mean() * 100)

