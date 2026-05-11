from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.common.paths import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config" / "dashboard.yml"


@st.cache_data(show_spinner=False)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_csv(relative_path: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"缺少展示数据文件：{relative_path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def add_error_columns(df: pd.DataFrame, phase_marks: dict[str, int]) -> pd.DataFrame:
    data = df.copy()
    data["error"] = data["simulated_price"] - data["actual_price"]
    data["abs_error"] = data["error"].abs()
    data["phase"] = "后期再定价"
    data.loc[data["day_index"] <= phase_marks["early_end_day"], "phase"] = "前期冲击"
    data.loc[
        (data["day_index"] > phase_marks["early_end_day"])
        & (data["day_index"] <= phase_marks["mid_end_day"]),
        "phase",
    ] = "中期平台"
    return data


def metric_value(df: pd.DataFrame, column: str) -> float:
    return float(df.iloc[0][column])


def price_figure(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    chart_config = config["charts"]
    price_band = config["price_band"]
    fig = go.Figure()
    fig.add_hrect(
        y0=price_band["lower"],
        y1=price_band["upper"],
        fillcolor=chart_config["price_band_color"],
        line_width=0,
        annotation_text="题面 110-120 美元/桶区间",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=df["trade_date"],
            y=df["actual_price"],
            mode="lines+markers",
            name="附件实际收盘价",
            line={"color": chart_config["actual_price_color"], "width": 2.4},
            marker={"size": 6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["trade_date"],
            y=df["simulated_price"],
            mode="lines+markers",
            name="短期动态模型",
            line={"color": chart_config["simulated_price_color"], "width": 2.4},
            marker={"size": 5},
        )
    )
    fig.update_layout(
        height=460,
        margin={"l": 20, "r": 20, "t": 42, "b": 20},
        legend={"orientation": "h", "y": 1.08},
        xaxis_title="日期",
        yaxis_title="美元/桶",
    )
    return fig


def error_figure(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    chart_config = config["charts"]
    colors = [
        chart_config["positive_error_color"] if value >= 0 else chart_config["negative_error_color"]
        for value in df["error"]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["trade_date"],
            y=df["error"],
            name="模型误差",
            marker_color=colors,
            customdata=df[["phase", "actual_price", "simulated_price"]],
            hovertemplate=(
                "日期=%{x}<br>阶段=%{customdata[0]}<br>"
                "误差=%{y:.2f}<br>实际=%{customdata[1]:.2f}<br>模拟=%{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color="#111827", line_width=1)
    fig.update_layout(
        height=330,
        margin={"l": 20, "r": 20, "t": 28, "b": 20},
        xaxis_title="日期",
        yaxis_title="模拟价 - 实际价",
        showlegend=False,
    )
    return fig


def mechanism_figure(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    premiums = config["mechanism_columns"]["premiums"]
    discounts = config["mechanism_columns"]["discounts"]
    labels = {
        "shortage_pressure": "供需缺口压力",
        "blockade_risk_premium": "封锁风险溢价",
        "uncertainty_premium": "不确定性溢价",
        "panic_premium": "恐慌溢价",
        "buffer_confirmation_discount": "缓冲确认折价",
        "expectation_relief_discount": "预期修复折价",
    }
    for column in premiums:
        if column in df.columns:
            fig.add_trace(go.Scatter(x=df["trade_date"], y=df[column], mode="lines", name=labels[column]))
    for column in discounts:
        if column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["trade_date"],
                    y=-df[column],
                    mode="lines",
                    name=labels[column],
                    line={"dash": "dash"},
                )
            )
    fig.add_hline(y=0, line_color="#111827", line_width=1)
    fig.update_layout(
        height=390,
        margin={"l": 20, "r": 20, "t": 34, "b": 20},
        legend={"orientation": "h", "y": 1.12},
        xaxis_title="日期",
        yaxis_title="价格贡献，正值推升，负值压低",
    )
    return fig


def segment_figure(segment_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=segment_df["分段"],
            y=segment_df["RMSE"],
            name="RMSE",
            marker_color="#2563eb",
            text=segment_df["RMSE"].round(2),
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=segment_df["分段"],
            y=segment_df["MAE"],
            name="MAE",
            marker_color="#0f766e",
            text=segment_df["MAE"].round(2),
            textposition="outside",
        )
    )
    fig.update_layout(
        barmode="group",
        height=390,
        margin={"l": 20, "r": 20, "t": 34, "b": 80},
        xaxis_title="误差分段",
        yaxis_title="美元/桶",
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def candidate_columns(candidates: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    columns = config["display_columns"]["candidate_table"]
    return [column for column in columns if column in candidates.columns]


def main() -> None:
    config = load_config()
    st.set_page_config(page_title=config["title"], layout="wide")

    st.title(config["title"])
    st.caption(config["subtitle"])

    paths = config["paths"]
    model_df = load_csv(paths["calibrated_path_csv"], parse_dates=["trade_date"])
    model_df = add_error_columns(model_df, config["phase_marks"])
    segment_df = load_csv(paths["segment_errors_csv"])
    candidates = load_csv(paths["top_candidates_csv"])
    best = candidates.iloc[[0]].copy()

    phase_options = ["全部", *model_df["phase"].drop_duplicates().tolist()]
    selected_phase = st.sidebar.selectbox("阶段筛选", phase_options)
    display_df = model_df if selected_phase == "全部" else model_df[model_df["phase"] == selected_phase]

    st.sidebar.markdown("### 资料入口")
    st.sidebar.code(paths["stage4_report"])
    st.sidebar.code(paths["reference_evidence"])

    metric_cols = st.columns(6)
    metric_cols[0].metric("RMSE", f"{metric_value(best, 'RMSE'):.2f}")
    metric_cols[1].metric("MAE", f"{metric_value(best, 'MAE'):.2f}")
    metric_cols[2].metric("峰值误差", f"{metric_value(best, '峰值误差'):.2f}")
    metric_cols[3].metric("末日误差", f"{metric_value(best, '末日误差'):.2f}")
    metric_cols[4].metric("后期 RMSE", f"{metric_value(best, '后期RMSE'):.2f}")
    metric_cols[5].metric("低价回落 RMSE", f"{metric_value(best, '低价回落RMSE'):.2f}")

    st.plotly_chart(price_figure(display_df, config), width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("误差时序")
        st.plotly_chart(error_figure(display_df, config), width="stretch")
    with right:
        st.subheader("分段误差")
        st.plotly_chart(segment_figure(segment_df), width="stretch")

    st.subheader("机制贡献分解")
    st.plotly_chart(mechanism_figure(display_df, config), width="stretch")

    st.subheader("候选参数对比")
    st.dataframe(candidates[candidate_columns(candidates, config)].round(4), width="stretch", hide_index=True)

    with st.expander("当前展示台读取的数据文件"):
        st.json(paths)


if __name__ == "__main__":
    main()
