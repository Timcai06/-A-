"""Build leakage-safe historical features for short-term ML audits.

The feature table is designed for one-step-ahead oil-price experiments.  Every
feature is shifted so it is available before the target trading day closes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent


INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "布伦特原油期货主力合约价格数据_清洗后.csv"
OUTPUT_CSV = PROJECT_ROOT / "output" / "calibration" / "短期机器学习特征样本.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "短期机器学习特征工程报告.md"


def rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    """Percentile of the latest available close within a rolling window."""

    def rank_last(arr: np.ndarray) -> float:
        if len(arr) == 0 or np.isnan(arr[-1]):
            return float("nan")
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return float("nan")
        return float((valid <= arr[-1]).mean())

    return values.rolling(window, min_periods=max(10, window // 3)).apply(rank_last, raw=True)


def build_feature_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    df = cleaned.copy().sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["pre_close_filled"] = df["pre_close"].fillna(df["close_price"].shift(1))
    df["target_log_return"] = np.log(df["close_price"] / df["pre_close_filled"])
    df["target_price_change"] = df["close_price"] - df["pre_close_filled"]
    df["target_close_price"] = df["close_price"]

    returns = df["target_log_return"]
    close = df["close_price"]
    prev_close = close.shift(1)

    for lag in [1, 2, 3, 5, 10]:
        df[f"滞后{lag}日收益率"] = returns.shift(lag)

    for window in [3, 5, 10, 20]:
        df[f"{window}日平均收益率"] = returns.shift(1).rolling(window, min_periods=max(2, window // 2)).mean()
        df[f"{window}日收益波动率"] = returns.shift(1).rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
        df[f"{window}日上涨天数占比"] = (
            (returns.shift(1) > 0).astype(float).rolling(window, min_periods=max(2, window // 2)).mean()
        )

    for window in [5, 20, 60]:
        ma = prev_close.rolling(window, min_periods=max(3, window // 3)).mean()
        df[f"相对{window}日均线偏离"] = prev_close / ma - 1

    df["前日振幅"] = ((df["high_price"] - df["low_price"]) / df["pre_close_filled"]).shift(1)
    df["前日开收盘偏离"] = ((df["close_price"] - df["open_price"]) / df["open_price"]).shift(1)
    df["60日价格分位"] = rolling_percentile(prev_close, 60)
    df["120日价格分位"] = rolling_percentile(prev_close, 120)
    df["是否冲突窗口"] = df["is_event_window"].astype(bool)

    keep_columns = [
        "trade_date",
        "pre_close_filled",
        "target_close_price",
        "target_log_return",
        "target_price_change",
        "是否冲突窗口",
    ] + [
        column
        for column in df.columns
        if column.startswith("滞后")
        or column.endswith("平均收益率")
        or column.endswith("收益波动率")
        or column.endswith("上涨天数占比")
        or column.startswith("相对")
        or column in {"前日振幅", "前日开收盘偏离", "60日价格分位", "120日价格分位"}
    ]
    return df[keep_columns].copy()


def build_report(features: pd.DataFrame) -> str:
    total_rows = len(features)
    usable_rows = int(features.dropna().shape[0])
    event_rows = int(features["是否冲突窗口"].sum())
    feature_count = len(features.columns) - 6
    start_date = pd.to_datetime(features["trade_date"]).min().date()
    end_date = pd.to_datetime(features["trade_date"]).max().date()

    return f"""# 短期机器学习特征工程报告

## 核心结论

已基于清洗后的 2017--2026 布伦特原油价格数据构造一日提前可用的机器学习特征样本。本表只使用价格自身的滞后信息，不引入任何无来源外生数据。

## 样本范围

| 项目 | 数值 |
|---|---:|
| 起始日期 | {start_date} |
| 截止日期 | {end_date} |
| 总行数 | {total_rows} |
| 完整可训练行数 | {usable_rows} |
| 冲突窗口行数 | {event_rows} |
| 特征数 | {feature_count} |

## 特征原则

- 所有收益率、波动率、均线和分位数特征都使用目标日之前的信息。
- 预测目标是当日收盘相对前收盘的对数收益率、价格变化和收盘价。
- 本特征表用于训练机器学习基准和残差修正模型，不替代综合机制递推主模型。

## 输出

- `{OUTPUT_CSV.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing cleaned data: {INPUT_CSV}")
    cleaned = pd.read_csv(INPUT_CSV, parse_dates=["trade_date"])
    features = build_feature_table(cleaned)
    ensure_parent(OUTPUT_CSV)
    ensure_parent(REPORT_PATH)
    features.to_csv(OUTPUT_CSV, index=False)
    REPORT_PATH.write_text(build_report(features), encoding="utf-8")

    print("Short-term ML feature table complete")
    print(f"Features: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
