"""Generate advanced Python figure candidates outside paper/figures.

These figures are visual experiments. They are intentionally written to
output/candidate_figures instead of paper/figures, because paper/figures should
only contain charts already cited by the final paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.paths import PROJECT_ROOT
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, SEABORN_PALETTE, configure_plot_style

OUT_DIR = PROJECT_ROOT / "output" / "candidate_figures"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "Python高级候选图表报告.md"
GREEN_BLUE_DIVERGING = LinearSegmentedColormap.from_list(
    "green_blue_diverging",
    [SCENARIO_COLORS["optimistic"], "#ffffff", SCENARIO_COLORS["fit"]],
)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, filename: str) -> Path:
    path = OUT_DIR / filename
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def load_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative)


def mechanism_waterfall(model: pd.DataFrame) -> Path:
    """Waterfall-style view of average mechanism pressure in the event window."""

    components = [
        ("供需缺口压力", float(model["shortage_pressure"].mean()), SCENARIO_COLORS["risk"]),
        ("封锁风险溢价", float(model["blockade_risk_premium"].mean()), SCENARIO_COLORS["risk"]),
        ("不确定性溢价", float(model["uncertainty_premium"].mean()), PAPER_COLORS["muted"]),
        ("恐慌溢价", float(model["panic_premium"].mean()), PAPER_COLORS["muted"]),
        ("缓冲确认折价", -float(model["buffer_confirmation_discount"].mean()), SCENARIO_COLORS["optimistic"]),
        ("预期修复折价", -float(model["expectation_relief_discount"].mean()), SCENARIO_COLORS["optimistic"]),
    ]

    labels = ["零基线"] + [item[0] for item in components] + ["净机制压力"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6.4))

    running = 0.0
    ax.bar(0, 0.0, color=SCENARIO_COLORS["actual"], width=0.62)
    ax.axhline(0, color=PAPER_COLORS["ink"], lw=1.0)

    for idx, (label, value, color) in enumerate(components, start=1):
        bottom = running if value >= 0 else running + value
        ax.bar(idx, abs(value), bottom=bottom, color=color, width=0.62, alpha=0.92)
        ax.plot([idx - 0.31, idx + 0.31], [running, running], color=PAPER_COLORS["border"], lw=1.0)
        sign = "+" if value >= 0 else ""
        text_y = bottom + abs(value) + (0.35 if value >= 0 else -0.8)
        ax.text(idx, text_y, f"{sign}{fmt(value)}", ha="center", fontsize=8.8)
        running += value

    ax.bar(len(labels) - 1, running, color=SCENARIO_COLORS["neutral"], width=0.62)
    ax.text(len(labels) - 1, running + 0.35, fmt(running), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("平均机制压力（美元/桶）")
    ax.set_title("短期冲突窗口的平均机制压力瀑布图")
    ax.text(
        0.01,
        0.98,
        "该图不把递推模型拆成价格恒等式，只展示各机制项在冲突窗口的平均方向和量级",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        color=PAPER_COLORS["muted"],
    )
    ax.grid(axis="y", alpha=0.45)
    return save(fig, "Python机制压力瀑布候选图.png")


def rolling_quality(rolling: pd.DataFrame) -> Path:
    rolling = rolling.copy()
    rolling["trade_date"] = pd.to_datetime(rolling["trade_date"])

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(rolling["trade_date"], rolling["rolling_rmse_7"], color=SCENARIO_COLORS["risk"], lw=2.2, label="7日滚动RMSE")
    ax.plot(rolling["trade_date"], rolling["rolling_mae_7"], color=SCENARIO_COLORS["neutral"], lw=2.0, label="7日滚动MAE")
    ax.plot(rolling["trade_date"], rolling["rolling_bias_7"].abs(), color=SCENARIO_COLORS["optimistic"], lw=1.8, label="7日绝对偏差")
    ax.fill_between(
        rolling["trade_date"],
        rolling["rolling_rmse_7"],
        color=SCENARIO_COLORS["risk"],
        alpha=0.08,
    )
    ax.set_title("短期模型滚动拟合质量")
    ax.set_ylabel("美元/桶")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月%d日"))
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.45)
    return save(fig, "Python短期滚动拟合质量候选图.png")


def error_heatmap(rolling: pd.DataFrame) -> Path:
    data = rolling.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["week"] = ((data["trade_date"] - data["trade_date"].min()).dt.days // 7).astype(int)
    data["weekday"] = data["trade_date"].dt.dayofweek
    pivot = data.pivot_table(index="weekday", columns="week", values="error", aggfunc="mean")
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
    im = ax.imshow(pivot, cmap=GREEN_BLUE_DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([weekdays[i] for i in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"第{int(col) + 1}周" for col in pivot.columns], rotation=0)
    ax.set_title("短期预测误差日历热力图")
    ax.set_xlabel("冲突窗口内周序")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("模型误差：模拟价 - 真实价（美元/桶）")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return save(fig, "Python短期误差日历热力候选图.png")


def sensitivity_heatmap(sensitivity: pd.DataFrame) -> Path:
    data = sensitivity[sensitivity["参数键"] != "baseline"].copy()
    top_keys = (
        data.groupby(["参数", "参数键"], as_index=False)["第180天价格相对基准变化"]
        .apply(lambda x: x.abs().max())
        .rename(columns={"第180天价格相对基准变化": "最大扰动"})
        .sort_values("最大扰动", ascending=False)
        .head(10)
    )
    data = data.merge(top_keys[["参数键"]], on="参数键", how="inner")
    data["扰动序号"] = data.groupby("参数键").cumcount() + 1
    data["参数"] = pd.Categorical(data["参数"], categories=list(reversed(top_keys["参数"].tolist())), ordered=True)
    pivot = data.pivot_table(index="参数", columns="扰动序号", values="第180天价格相对基准变化", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    vmax = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
    im = ax.imshow(pivot, cmap=GREEN_BLUE_DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("长期敏感性扰动热力图")
    ax.set_xlabel("参数扰动档位")
    ax.set_ylabel(None)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"档位{int(col)}" for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("第180天价格相对基准变化（美元/桶）")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return save(fig, "Python长期敏感性热力候选图.png")


def feature_correlation_heatmap(features: pd.DataFrame) -> Path:
    """Seaborn heatmap for leakage-safe historical feature relationships."""

    columns = [
        "target_log_return",
        "滞后1日收益率",
        "滞后2日收益率",
        "滞后5日收益率",
        "5日平均收益率",
        "5日收益波动率",
        "10日收益波动率",
        "20日收益波动率",
        "相对20日均线偏离",
        "相对60日均线偏离",
        "前日振幅",
        "60日价格分位",
    ]
    labels = {
        "target_log_return": "目标收益",
        "滞后1日收益率": "滞后1日",
        "滞后2日收益率": "滞后2日",
        "滞后5日收益率": "滞后5日",
        "5日平均收益率": "5日均值",
        "5日收益波动率": "5日波动",
        "10日收益波动率": "10日波动",
        "20日收益波动率": "20日波动",
        "相对20日均线偏离": "偏离20日线",
        "相对60日均线偏离": "偏离60日线",
        "前日振幅": "前日振幅",
        "60日价格分位": "60日分位",
    }
    usable = features[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    corr = usable.corr(method="spearman").rename(index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(10.2, 8.2))
    sns.heatmap(
        corr,
        ax=ax,
        cmap=GREEN_BLUE_DIVERGING,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"shrink": 0.76, "label": "Spearman相关系数"},
    )
    ax.set_title("滞后特征相关性热力图")
    ax.tick_params(axis="x", rotation=35, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0, labelsize=8.5)
    ax.text(
        0.0,
        -0.13,
        "所有特征均为目标日前可观测变量，用于检查机器学习基准是否存在高度冗余输入。",
        transform=ax.transAxes,
        fontsize=9.2,
        color=PAPER_COLORS["muted"],
    )
    return save(fig, "Seaborn滞后特征相关性热力候选图.png")


def segmented_residual_distribution(rolling: pd.DataFrame) -> Path:
    """Seaborn distribution view for residuals across short-term event phases."""

    data = rolling.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["阶段"] = pd.cut(
        data["day_index"],
        bins=[-1, 10, 28, 10_000],
        labels=["冲击启动期", "高位平台期", "回落修复期"],
    )
    data["绝对误差"] = data["error"].abs()

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sns.violinplot(
        data=data,
        x="阶段",
        y="error",
        hue="阶段",
        dodge=False,
        palette=SEABORN_PALETTE[:3],
        inner=None,
        linewidth=0.8,
        saturation=0.9,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=data,
        x="阶段",
        y="error",
        hue="阶段",
        dodge=False,
        width=0.22,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": PAPER_COLORS["ink"], "linewidth": 0.9, "alpha": 0.88},
        whiskerprops={"color": PAPER_COLORS["ink"], "linewidth": 0.9},
        medianprops={"color": SCENARIO_COLORS["fit"], "linewidth": 1.4},
        showfliers=False,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="阶段",
        y="error",
        color=PAPER_COLORS["ink"],
        alpha=0.42,
        size=3.2,
        jitter=0.16,
        ax=ax,
    )
    ax.axhline(0, color=PAPER_COLORS["ink"], linewidth=1.0)
    ax.set_title("短期模型残差的分段分布")
    ax.set_xlabel(None)
    ax.set_ylabel("模型误差：模拟价 - 真实价（美元/桶）")
    phase_mae = data.groupby("阶段", observed=False)["绝对误差"].mean()
    for idx, (phase, value) in enumerate(phase_mae.items()):
        ax.text(idx, data["error"].min() - 0.45, f"MAE={value:.2f}", ha="center", fontsize=9.2, color=PAPER_COLORS["muted"])
    ax.grid(axis="y", alpha=0.42)
    return save(fig, "Seaborn短期残差分段分布候选图.png")


def candidate_tradeoff(candidates: pd.DataFrame) -> Path:
    selected = candidates.head(10).copy()
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    source_colors = {
        "local_stability_refinement": SCENARIO_COLORS["neutral"],
        "fit_quality_refinement": SCENARIO_COLORS["optimistic"],
        "continuous_refinement": PAPER_COLORS["navy"],
        "seeded_random": PAPER_COLORS["muted"],
    }
    colors = [source_colors.get(str(item), PAPER_COLORS["muted"]) for item in selected["candidate_source"]]
    sizes = 90 + selected["低价回落RMSE"].rank(pct=True).to_numpy() * 260
    ax.scatter(
        selected["RMSE"],
        selected["高价平台RMSE"],
        s=sizes,
        c=colors,
        alpha=0.82,
        edgecolor="white",
        linewidth=1.1,
    )
    offsets = [(-0.045, 0.035), (-0.02, -0.08), (0.025, 0.045), (0.025, 0.02), (0.025, -0.02)]
    for idx, (_, row) in enumerate(selected.iterrows()):
        dx, dy = offsets[idx % len(offsets)]
        ax.text(row["RMSE"] + dx, row["高价平台RMSE"] + dy, str(int(row["candidate_id"])), fontsize=8.5)
    ax.set_title("短期候选模型的精度与平台解释权衡")
    ax.set_xlabel("整体 RMSE（越低越好）")
    ax.set_ylabel("高价平台 RMSE（越低越好）")
    ax.text(
        0.02,
        0.98,
        "点越大表示低价回落段误差越高；理想候选位于左下角且点较小",
        transform=ax.transAxes,
        va="top",
        fontsize=9.2,
        color=PAPER_COLORS["muted"],
    )
    ax.grid(alpha=0.45)
    return save(fig, "Python候选模型权衡气泡候选图.png")


def main() -> None:
    ensure_dirs()
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=14)

    paths: list[Path] = []
    model = load_csv("output/calibration/动态模型校准后路径.csv")
    rolling = load_csv("output/calibration/短期模型滚动误差.csv")
    sensitivity = load_csv("output/sensitivity/敏感性分析结果.csv")
    candidates = load_csv("output/calibration/动态模型候选参数前10.csv")
    features = load_csv("output/calibration/短期机器学习特征样本.csv")

    paths.append(mechanism_waterfall(model))
    paths.append(rolling_quality(rolling))
    paths.append(error_heatmap(rolling))
    paths.append(sensitivity_heatmap(sensitivity))
    paths.append(candidate_tradeoff(candidates))
    paths.append(feature_correlation_heatmap(features))
    paths.append(segmented_residual_distribution(rolling))

    report = [
        "# Python高级候选图表报告",
        "",
        "## 输出规则",
        "",
        "本轮图表均为候选图，未进入论文引用前只保存在 `output/candidate_figures/`，不放入 `paper/figures/`。",
        "",
        "## 候选图清单",
        "",
    ]
    for path in paths:
        report.append(f"- `{path.relative_to(PROJECT_ROOT)}`")
    report.extend(
        [
            "",
            "## 初步判断",
            "",
            "- 机制贡献瀑布图适合解释短期峰值价格由哪些机制项推高或压低。",
            "- 滚动拟合质量图适合回答短期模型是否只在局部窗口表现较好。",
            "- 误差日历热力图适合快速暴露误差集中在哪几周和哪些交易日。",
            "- 长期敏感性热力图适合替代单一龙卷风图，展示不同扰动档位下的价格方向。",
            "- 候选模型权衡气泡图适合说明主模型选择不是单一 RMSE 排序，而是多目标权衡。",
            "- Seaborn 滞后特征相关性热力图适合说明历史机器学习特征的冗余结构和信息边界。",
            "- Seaborn 短期残差分段分布图适合检查冲击启动、高位平台和回落修复三段的误差偏态。",
        ]
    )
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Generated candidate figures:")
    for path in paths:
        print(path)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
