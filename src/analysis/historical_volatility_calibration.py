"""Use 2017-2025 price history to calibrate model uncertainty.

The historical Brent series is not used to overwrite the Hormuz mechanism.
Instead, it supplies empirical scales for long-horizon uncertainty:
- daily disturbance in the state-transition scenario tree;
- stress noise in Monte Carlo parameter sampling;
- jump-size scaling for escalation/easing state switches.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.metrics import rmse
from src.common.paths import PROJECT_ROOT, ensure_parent
from src.common.plotting import configure_plot_style


EVENT_START = pd.Timestamp("2026-03-02")
WINDOW_SIZE = 46
PRICE_CSV = PROJECT_ROOT / "data" / "processed" / "布伦特原油期货主力合约价格数据_清洗后.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "history"
PARAMETER_CSV = OUTPUT_DIR / "历史波动参数校准.csv"
WINDOW_CSV = OUTPUT_DIR / "历史波动校准窗口.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "历史数据模型参数增强报告.md"
FIGURE_PATH = PROJECT_ROOT / "figures" / "历史波动参数校准.png"


@dataclass(frozen=True)
class HistoricalModelFactors:
    state_daily_sigma: float
    monte_carlo_stress_noise_sigma: float
    transition_jump_sigma_scale: float
    event_volatility_percentile: float
    event_max_jump_percentile: float


DEFAULT_FACTORS = HistoricalModelFactors(
    state_daily_sigma=0.0090,
    monte_carlo_stress_noise_sigma=0.060,
    transition_jump_sigma_scale=1.0,
    event_volatility_percentile=0.95,
    event_max_jump_percentile=0.95,
)


def load_clean_prices() -> pd.DataFrame:
    if not PRICE_CSV.exists():
        raise FileNotFoundError(f"Missing cleaned price CSV: {PRICE_CSV}")
    df = pd.read_csv(PRICE_CSV, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    df["close_price"] = df["close_price"].astype(float)
    df["pre_close"] = df["pre_close"].astype(float)
    df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1))
    return df


def percentile_rank(reference: pd.Series, value: float) -> float:
    reference = reference.dropna()
    if reference.empty:
        return float("nan")
    return float((reference <= value).mean())


def window_metrics(window: pd.DataFrame) -> dict[str, float | str]:
    close = window["close_price"].astype(float).reset_index(drop=True)
    returns = np.log(close / close.shift(1)).dropna()
    first_pre_close = float(window["pre_close"].iloc[0])
    naive = pd.concat([pd.Series([first_pre_close]), close.iloc[:-1]], ignore_index=True)
    error = close - naive
    return {
        "窗口起始": window["trade_date"].min().date().isoformat(),
        "窗口终止": window["trade_date"].max().date().isoformat(),
        "日收益波动率": float(returns.std(ddof=0)),
        "实现波动率_46日": float(returns.std(ddof=0) * np.sqrt(len(window))),
        "最大单日绝对收益": float(returns.abs().max()),
        "平均绝对日收益": float(returns.abs().mean()),
        "累计收益率": float(close.iloc[-1] / close.iloc[0] - 1),
        "价格区间比例": float((close.max() - close.min()) / close.iloc[0]),
        "朴素上一日基准_RMSE": rmse(error),
    }


def build_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str]]:
    reference = df[df["trade_date"] < EVENT_START].copy().reset_index(drop=True)
    event = df[df["trade_date"] >= EVENT_START].head(WINDOW_SIZE).copy()
    if len(event) < WINDOW_SIZE:
        raise RuntimeError("Event window is shorter than required calibration window.")
    rows = [
        window_metrics(reference.iloc[start : start + WINDOW_SIZE])
        for start in range(0, len(reference) - WINDOW_SIZE + 1)
    ]
    return pd.DataFrame(rows), window_metrics(event)


def build_parameter_table(windows: pd.DataFrame, event_metrics: dict[str, float | str]) -> pd.DataFrame:
    event_daily_sigma = float(event_metrics["日收益波动率"])
    event_max_jump = float(event_metrics["最大单日绝对收益"])
    hist_daily_p75 = float(windows["日收益波动率"].quantile(0.75))
    hist_daily_p90 = float(windows["日收益波动率"].quantile(0.90))
    hist_jump_p95 = float(windows["最大单日绝对收益"].quantile(0.95))
    vol_percentile = percentile_rank(windows["日收益波动率"], event_daily_sigma)
    jump_percentile = percentile_rank(windows["最大单日绝对收益"], event_max_jump)

    state_daily_sigma = float(np.clip((0.60 * event_daily_sigma + 0.40 * hist_daily_p90) * 0.35, 0.006, 0.014))
    stress_noise_sigma = float(np.clip(0.040 + 0.035 * vol_percentile, 0.040, 0.080))
    jump_scale = float(np.clip(event_max_jump / max(hist_jump_p95, 1e-6), 0.80, 1.60))

    rows = [
        ("历史日收益波动率P75", hist_daily_p75, "log return/day", "2017-2025同长度窗口", "衡量普通高波动窗口"),
        ("历史日收益波动率P90", hist_daily_p90, "log return/day", "2017-2025同长度窗口", "长期状态扰动参照"),
        ("历史最大单日跳变P95", hist_jump_p95, "abs log return/day", "2017-2025同长度窗口", "状态切换跳变参照"),
        ("冲突窗口日收益波动率", event_daily_sigma, "log return/day", "2026冲突窗口", "事件强度参照"),
        ("冲突窗口最大单日跳变", event_max_jump, "abs log return/day", "2026冲突窗口", "事件跳变强度参照"),
        ("冲突窗口波动历史分位数", vol_percentile, "percentile", "历史窗口对比", "判断事件是否高压"),
        ("冲突窗口跳变历史分位数", jump_percentile, "percentile", "历史窗口对比", "判断事件是否高跳变"),
        ("状态转移日度扰动波动率", state_daily_sigma, "price return scale", "历史+事件混合校准", "进入长期状态转移模型"),
        ("蒙特卡洛压力噪声标准差", stress_noise_sigma, "stress index scale", "历史波动分位校准", "进入蒙特卡洛压力抽样"),
        ("状态切换跳变标准差乘数", jump_scale, "multiplier", "历史跳变P95校准", "进入长期状态转移模型"),
    ]
    return pd.DataFrame(rows, columns=["参数", "数值", "单位", "来源", "用途"])


def factors_from_table(table: pd.DataFrame) -> HistoricalModelFactors:
    values = dict(zip(table["参数"], table["数值"], strict=False))
    return HistoricalModelFactors(
        state_daily_sigma=float(values.get("状态转移日度扰动波动率", DEFAULT_FACTORS.state_daily_sigma)),
        monte_carlo_stress_noise_sigma=float(
            values.get("蒙特卡洛压力噪声标准差", DEFAULT_FACTORS.monte_carlo_stress_noise_sigma)
        ),
        transition_jump_sigma_scale=float(values.get("状态切换跳变标准差乘数", DEFAULT_FACTORS.transition_jump_sigma_scale)),
        event_volatility_percentile=float(values.get("冲突窗口波动历史分位数", DEFAULT_FACTORS.event_volatility_percentile)),
        event_max_jump_percentile=float(values.get("冲突窗口跳变历史分位数", DEFAULT_FACTORS.event_max_jump_percentile)),
    )


def load_historical_model_factors() -> HistoricalModelFactors:
    if not PARAMETER_CSV.exists():
        return DEFAULT_FACTORS
    return factors_from_table(pd.read_csv(PARAMETER_CSV))


def save_figure(windows: pd.DataFrame, event_metrics: dict[str, float | str]) -> None:
    configure_plot_style()
    ensure_parent(FIGURE_PATH)
    event_sigma = float(event_metrics["日收益波动率"])
    event_jump = float(event_metrics["最大单日绝对收益"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].hist(windows["日收益波动率"], bins=36, color="#93c5fd", edgecolor="white")
    axes[0].axvline(event_sigma, color="#dc2626", linewidth=2.0, label="2026冲突窗口")
    axes[0].set_title("历史46日窗口日收益波动率分布")
    axes[0].set_xlabel("日收益波动率")
    axes[0].set_ylabel("窗口数量")
    axes[0].legend()

    axes[1].hist(windows["最大单日绝对收益"], bins=36, color="#bbf7d0", edgecolor="white")
    axes[1].axvline(event_jump, color="#dc2626", linewidth=2.0, label="2026冲突窗口")
    axes[1].set_title("历史46日窗口最大单日跳变分布")
    axes[1].set_xlabel("最大单日绝对收益")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def build_report(table: pd.DataFrame, windows: pd.DataFrame, event_metrics: dict[str, float | str]) -> str:
    factors = factors_from_table(table)
    table_md = table.to_markdown(index=False, floatfmt=".4f")
    return f"""# 历史数据模型参数增强报告

## 结论摘要

本报告把 2017-2025 的附件历史价格数据转化为长期模型的不确定性参数。历史数据不直接替代霍尔木兹封锁机制，也不反向拟合 2026 年价格；它只用于校准长期状态扰动、蒙特卡洛压力噪声和状态切换跳变尺度。

- 历史同长度窗口数量：{len(windows)}
- 冲突窗口日收益波动率历史分位数：{factors.event_volatility_percentile:.1%}
- 冲突窗口最大单日跳变历史分位数：{factors.event_max_jump_percentile:.1%}
- 状态转移日度扰动波动率：{factors.state_daily_sigma:.4f}
- 蒙特卡洛压力噪声标准差：{factors.monte_carlo_stress_noise_sigma:.4f}
- 状态切换跳变标准差乘数：{factors.transition_jump_sigma_scale:.2f}

## 参数表

{table_md}

## 建模含义

2017-2025 数据对短期主模型的作用主要是基准和稳健性检验；对长期模型的作用更直接：它提供历史波动和跳变尺度，使长期预测不再只是三条光滑中心线，而是带有历史依据的条件区间和尾部风险。

## 输出产物

- `{PARAMETER_CSV.relative_to(PROJECT_ROOT)}`
- `{WINDOW_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""


def write_outputs(windows: pd.DataFrame, table: pd.DataFrame, event_metrics: dict[str, float | str]) -> None:
    ensure_parent(PARAMETER_CSV)
    ensure_parent(WINDOW_CSV)
    ensure_parent(REPORT_PATH)
    windows.to_csv(WINDOW_CSV, index=False)
    table.to_csv(PARAMETER_CSV, index=False)
    save_figure(windows, event_metrics)
    REPORT_PATH.write_text(build_report(table, windows, event_metrics), encoding="utf-8")


def main() -> None:
    df = load_clean_prices()
    windows, event_metrics = build_windows(df)
    table = build_parameter_table(windows, event_metrics)
    write_outputs(windows, table, event_metrics)
    print("Historical volatility calibration complete")
    print(f"Parameters: {PARAMETER_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
