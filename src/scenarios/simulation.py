"""Long-horizon scenario forecast simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models import dynamic_short_term as dynamic
from src.scenarios.settings import CALIBRATED_PATH, FORECAST_END_DAY, MARKER_DAYS, SCENARIO_NAMES


# Long-horizon mechanism coefficients.
#
# These are not physical constants. They translate the short-term calibrated
# mechanism into the 60-180 day scenario horizon, where SPR releases taper,
# early shock uncertainty decays, and unresolved blockade risk persists. They
# stay at module level because the sensitivity module deliberately perturbs
# them one at a time without changing the calibrated short-term dataclasses.
GAP_CLOSURE_SHARE = 0.005
SPR_PRICE_STRESS_START_RATIO = 1.03
SPR_PRICE_STRESS_WIDTH = 0.25
SPR_MIN_POLICY_SHARE = 0.08
SPR_PRICE_POLICY_SHARE = 0.16
SPR_TAPER_START_DAY = 75
SPR_TAPER_FLOOR = 0.25
SPR_TAPER_DECAY_DAYS = 45

UNCERTAINTY_BUILDUP_DAYS = 18
SHOCK_UNCERTAINTY_SHARE = 0.45
SHOCK_DECAY_START_DAY = 45
SHOCK_DECAY_DAYS = 70
REGIME_RISK_SHARE = 0.90
REGIME_BASE_SHARE = 0.20
REGIME_STRESS_SHARE = 0.80
REGIME_CONFIDENCE_DECAY_START_DAY = 60
REGIME_CONFIDENCE_DECAY_FLOOR = 0.40
REGIME_CONFIDENCE_DECAY_DAYS = 120

PANIC_PRICE_MULTIPLIER = 0.45
FEAR_CHANGE_MOMENTUM = 2.5
OVERSUPPLY_REVERSION_SCALE = 1.35
BLOCKADE_RISK_DECAY = dynamic.BLOCKADE_RISK_DECAY


@dataclass(frozen=True)
class MechanismConstantNote:
    name: str
    value: float
    role: str
    basis: str
    audit_status: str


MECHANISM_CONSTANT_NOTES: tuple[MechanismConstantNote, ...] = (
    MechanismConstantNote(
        "SPR_TAPER_START_DAY",
        SPR_TAPER_START_DAY,
        "SPR 由计划释放转向缺口驱动收缩的起点",
        "60 天后进入长期外推，75 天表示政策释放开始从应急高强度转向库存保护；不是题面物理常数。",
        "已纳入敏感性分析",
    ),
    MechanismConstantNote(
        "SPR_TAPER_FLOOR",
        SPR_TAPER_FLOOR,
        "长期仍保留的最低政策释放比例",
        "用于避免 SPR 在制度风险仍存在时立即归零，表示保守释放底座。",
        "暂作为机制设定",
    ),
    MechanismConstantNote(
        "REGIME_CONFIDENCE_DECAY_FLOOR",
        REGIME_CONFIDENCE_DECAY_FLOOR,
        "市场信心恢复后仍保留的制度风险比例",
        "表示封锁未彻底解除时，航运保险、安全和再升级风险仍有残余溢价。",
        "已纳入敏感性分析",
    ),
    MechanismConstantNote(
        "OVERSUPPLY_REVERSION_SCALE",
        OVERSUPPLY_REVERSION_SCALE,
        "供给过剩对目标价格的向下回归强度",
        "刻画长期轻微过剩对价格的保守下拉，不作为供需弹性替代。",
        "已纳入敏感性分析",
    ),
    MechanismConstantNote(
        "FEAR_CHANGE_MOMENTUM",
        FEAR_CHANGE_MOMENTUM,
        "恐慌变化对日度价格的短期动量修正",
        "作为事件冲击吸收速度项，只影响恐慌变化率；论文中应表述为辅助动量审计点。",
        "保留但需审慎解释",
    ),
    MechanismConstantNote(
        "BLOCKADE_RISK_DECAY",
        BLOCKADE_RISK_DECAY,
        "封锁风险溢价的日度衰减速度",
        "刻画封锁风险从冲击确认到制度化吸收的衰减速度；不是题面物理常数。",
        "已纳入敏感性分析",
    ),
)


def build_forecast_frame(event_df: pd.DataFrame) -> pd.DataFrame:
    event_start = event_df["trade_date"].min()
    last_day_index = int((event_df["trade_date"].max() - event_start).days)
    future_dates = pd.date_range(
        event_df["trade_date"].max() + pd.Timedelta(days=1),
        event_start + pd.Timedelta(days=FORECAST_END_DAY),
        freq="D",
    )
    future_rows = pd.DataFrame(
        {
            "trade_date": future_dates,
            "close_price": np.nan,
            "pre_close": np.nan,
        }
    )
    frame = pd.concat([event_df[["trade_date", "close_price", "pre_close"]], future_rows], ignore_index=True)
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    frame["阶段"] = np.where(
        frame["trade_date"] <= event_df["trade_date"].max(),
        "附件观测期",
        "情景外推期",
    )
    frame["day_index"] = (frame["trade_date"] - event_start).dt.days.astype(int)
    frame.loc[frame.index[0], "pre_close"] = float(event_df.iloc[0]["pre_close"])
    if int(frame["day_index"].max()) != FORECAST_END_DAY:
        raise RuntimeError("Forecast frame did not reach the configured end day.")
    if last_day_index >= FORECAST_END_DAY:
        raise RuntimeError("Observed event window already exceeds forecast horizon.")
    return frame


def load_calibrated_prefix() -> pd.DataFrame:
    if not CALIBRATED_PATH.exists():
        raise FileNotFoundError(f"Missing Stage 4 calibrated path: {CALIBRATED_PATH}")
    prefix = pd.read_csv(CALIBRATED_PATH, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    prefix["阶段"] = "附件观测期"
    prefix["is_observed_price"] = True
    prefix["forecast_price"] = prefix["simulated_price"]
    return prefix


def infer_gap_closure_day(prefix: pd.DataFrame, assumptions: dynamic.PhysicalAssumptions) -> int | None:
    closed = prefix[prefix["supply_gap"] <= assumptions.base_demand * GAP_CLOSURE_SHARE]
    if closed.empty:
        return None
    return int(closed.iloc[0]["day_index"])


def adaptive_spr_release(
    day_index: int,
    scheduled_spr: float,
    gross_gap_before_spr: float,
    previous_price: float,
    base_price: float,
    assumptions: dynamic.PhysicalAssumptions,
) -> tuple[float, float]:
    """Taper SPR releases once the physical shortage is mostly covered.

    The release is demand-led instead of schedule-led: when the remaining
    physical gap closes, planned SPR capacity is not mechanically exhausted.
    """
    if scheduled_spr <= 0:
        return 0.0, 0.0

    reserve_buffer = assumptions.base_demand * GAP_CLOSURE_SHARE
    coverage_need = max(gross_gap_before_spr + reserve_buffer, 0.0)
    demand_based_release = min(scheduled_spr, coverage_need)

    stress_ratio = np.clip(gross_gap_before_spr / max(assumptions.supply_interruption, 1.0), 0.0, 1.0)
    price_stress = np.clip(
        (previous_price / base_price - SPR_PRICE_STRESS_START_RATIO) / SPR_PRICE_STRESS_WIDTH,
        0.0,
        1.0,
    )
    minimum_policy_share = SPR_MIN_POLICY_SHARE + SPR_PRICE_POLICY_SHARE * price_stress
    stress_share = max(minimum_policy_share, stress_ratio)

    time_taper = 1.0
    if day_index > SPR_TAPER_START_DAY:
        time_taper = SPR_TAPER_FLOOR + (1 - SPR_TAPER_FLOOR) * np.exp(
            -(day_index - SPR_TAPER_START_DAY) / SPR_TAPER_DECAY_DAYS
        )

    tapered_release = scheduled_spr * stress_share * time_taper
    actual_release = float(max(demand_based_release, tapered_release))
    actual_release = float(min(scheduled_spr, actual_release))
    taper_ratio = actual_release / scheduled_spr
    return actual_release, taper_ratio


def uncertainty_components(
    day_index: int,
    base_price: float,
    behavior: dynamic.BehavioralParameters,
    assumptions: dynamic.PhysicalAssumptions,
    spr_release: float,
    route_supply: float,
    demand_decline: float,
) -> tuple[float, float, float]:
    """Split early shock uncertainty from persistent blockade-regime risk.

    Regime risk is still allowed to persist, but it now depends on unresolved
    physical stress and fades as the market observes buffers working. This
    avoids a constant long-run uncertainty floor after the shortage is covered.
    """
    buildup = 1 - np.exp(-day_index / UNCERTAINTY_BUILDUP_DAYS)
    shock_decay = np.exp(-max(day_index - SHOCK_DECAY_START_DAY, 0) / SHOCK_DECAY_DAYS)
    shock_uncertainty = base_price * behavior.uncertainty_floor * SHOCK_UNCERTAINTY_SHARE * buildup * shock_decay

    unresolved_stress = (
        assumptions.supply_interruption
        - spr_release
        - route_supply
        - demand_decline
    ) / max(assumptions.supply_interruption, 1.0)
    regime_share = REGIME_BASE_SHARE + REGIME_STRESS_SHARE * np.clip(unresolved_stress, 0.0, 1.0)
    confidence_decay = 1.0
    if day_index > REGIME_CONFIDENCE_DECAY_START_DAY:
        confidence_decay = REGIME_CONFIDENCE_DECAY_FLOOR + (1 - REGIME_CONFIDENCE_DECAY_FLOOR) * np.exp(
            -(day_index - REGIME_CONFIDENCE_DECAY_START_DAY) / REGIME_CONFIDENCE_DECAY_DAYS
        )
    regime_risk = base_price * behavior.uncertainty_floor * REGIME_RISK_SHARE * regime_share * confidence_decay

    return float(shock_uncertainty + regime_risk), float(shock_uncertainty), float(regime_risk)


def oversupply_discount(
    oversupply: float,
    base_price: float,
    elasticity: float,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> float:
    """Allow persistent excess supply to pull the target price downward."""
    if oversupply <= 0:
        return 0.0
    return float(
        base_price
        * behavior.pressure_scale
        * OVERSUPPLY_REVERSION_SCALE
        * (oversupply / assumptions.base_demand)
        / max(abs(elasticity), 0.01)
    )


def simulate_future_from_prefix(
    prefix: pd.DataFrame,
    future_frame: pd.DataFrame,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
    base_price: float,
) -> pd.DataFrame:
    previous_price = float(prefix.iloc[-1]["simulated_price"])
    previous_day = int(prefix.iloc[-1]["day_index"])
    previous_fear_excess = assumptions.fear_initial * np.exp(-assumptions.fear_decay * previous_day)
    inventory_remaining = float(prefix.iloc[-1]["inventory_remaining"])
    gap_closure_day = infer_gap_closure_day(prefix, assumptions)
    rows: list[dict[str, Any]] = []

    for _, row in future_frame.iterrows():
        trade_date = row["trade_date"]
        day_index = int(row["day_index"])
        elasticity = dynamic.interpolate_elasticity(day_index, assumptions)
        price_ratio = max(previous_price / base_price, 0.1)

        price_adjusted_demand = assumptions.base_demand * (price_ratio**elasticity)
        demand_decline = dynamic.ramp(
            day_index,
            0,
            assumptions.demand_decline_ramp_days,
            assumptions.observed_demand_decline,
        )
        effective_demand = max(price_adjusted_demand - demand_decline, assumptions.base_demand * 0.70)

        scheduled_spr_release = dynamic.ramp(
            day_index,
            assumptions.spr_delay_days,
            assumptions.spr_ramp_days,
            assumptions.spr_max_release,
        )
        route_supply = dynamic.ramp(
            day_index,
            assumptions.route_start_day,
            assumptions.route_ramp_days,
            assumptions.route_max_capacity,
        )
        supply_before_spr = assumptions.base_supply - assumptions.supply_interruption + route_supply
        gross_gap_before_spr = max(effective_demand - supply_before_spr, 0.0)
        spr_release, spr_taper_ratio = adaptive_spr_release(
            day_index,
            scheduled_spr_release,
            gross_gap_before_spr,
            previous_price,
            base_price,
            assumptions,
        )
        supply_without_inventory = supply_before_spr + spr_release
        raw_balance = effective_demand - supply_without_inventory
        raw_gap = max(raw_balance, 0.0)

        inventory_buffer = min(
            raw_gap * behavior.inventory_response,
            assumptions.inventory_daily_cap,
            inventory_remaining,
        )
        inventory_remaining -= inventory_buffer
        effective_supply = supply_without_inventory + inventory_buffer
        supply_balance = effective_supply - effective_demand
        residual_gap = max(-supply_balance, 0.0)
        oversupply = max(supply_balance, 0.0)
        if gap_closure_day is None and residual_gap <= assumptions.base_demand * GAP_CLOSURE_SHARE:
            gap_closure_day = day_index

        fear_excess = assumptions.fear_initial * np.exp(-assumptions.fear_decay * day_index)
        shortage_pressure = (
            base_price
            * behavior.pressure_scale
            * (residual_gap / assumptions.base_demand)
            / max(abs(elasticity), 0.01)
        )
        blockade_risk_premium = (
            base_price
            * behavior.risk_weight
            * (assumptions.supply_interruption / assumptions.base_demand)
            * (1 - np.exp(-day_index / 7))
            * np.exp(-BLOCKADE_RISK_DECAY * day_index)
        )
        uncertainty_premium, shock_uncertainty_premium, regime_risk_premium = uncertainty_components(
            day_index,
            base_price,
            behavior,
            assumptions,
            spr_release,
            route_supply,
            demand_decline,
        )
        panic_premium = base_price * PANIC_PRICE_MULTIPLIER * fear_excess
        buffer_discount = dynamic.buffer_confirmation_discount(day_index, gap_closure_day, base_price, behavior)
        relief_discount = dynamic.expectation_relief_discount(day_index, base_price, behavior)
        excess_supply_discount = oversupply_discount(oversupply, base_price, elasticity, assumptions, behavior)
        target_price = (
            base_price
            + shortage_pressure
            + blockade_risk_premium
            + uncertainty_premium
            + panic_premium
            - buffer_discount
            - relief_discount
            - excess_supply_discount
        )

        simulated_price = previous_price + behavior.adjustment_speed * (target_price - previous_price)
        simulated_price += FEAR_CHANGE_MOMENTUM * (fear_excess - previous_fear_excess)
        simulated_price = float(np.clip(simulated_price, base_price * 0.75, 180.0))

        rows.append(
            {
                "day_index": day_index,
                "trade_date": trade_date,
                "actual_price": np.nan,
                "simulated_price": simulated_price,
                "effective_supply": effective_supply,
                "effective_demand": effective_demand,
                "supply_gap": residual_gap,
                "supply_balance": supply_balance,
                "gross_supply_gap_before_spr": gross_gap_before_spr,
                "scheduled_spr_release": scheduled_spr_release,
                "spr_release": spr_release,
                "spr_taper_ratio": spr_taper_ratio,
                "route_supply": route_supply,
                "inventory_buffer": inventory_buffer,
                "inventory_remaining": inventory_remaining,
                "demand_decline": demand_decline,
                "demand_elasticity": elasticity,
                "fear_factor": 1 + fear_excess,
                "shortage_pressure": shortage_pressure,
                "blockade_risk_premium": blockade_risk_premium,
                "uncertainty_premium": uncertainty_premium,
                "shock_uncertainty_premium": shock_uncertainty_premium,
                "regime_risk_premium": regime_risk_premium,
                "panic_premium": panic_premium,
                "buffer_confirmation_discount": buffer_discount,
                "expectation_relief_discount": relief_discount,
                "oversupply_discount": excess_supply_discount,
                "阶段": "情景外推期",
                "is_observed_price": False,
                "forecast_price": simulated_price,
            }
        )
        previous_price = simulated_price
        previous_fear_excess = fear_excess

    return pd.DataFrame(rows)


def run_scenario(
    scenario_key: str,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
    forecast_frame: pd.DataFrame,
    prefix: pd.DataFrame,
) -> pd.DataFrame:
    future_frame = forecast_frame[forecast_frame["阶段"] == "情景外推期"].copy()
    base_price = float(forecast_frame.iloc[0]["pre_close"])
    future = simulate_future_from_prefix(prefix, future_frame, assumptions, behavior, base_price)
    simulation = pd.concat([prefix, future], ignore_index=True)
    simulation["scenario"] = scenario_key
    simulation["情景"] = SCENARIO_NAMES[scenario_key]
    return simulation


def summarize_scenario(simulation: pd.DataFrame) -> dict[str, Any]:
    scenario_key = str(simulation.iloc[0]["scenario"])
    forecast_period = simulation[simulation["阶段"] == "情景外推期"]
    all_period = simulation.copy()
    marker_prices = {
        f"第{day}天价格": float(all_period.loc[all_period["day_index"] == day, "forecast_price"].iloc[0])
        for day in MARKER_DAYS
    }
    post_observed_peak = float(forecast_period["forecast_price"].max())
    post_observed_min = float(forecast_period["forecast_price"].min())
    final_price = float(all_period.loc[all_period["day_index"] == FORECAST_END_DAY, "forecast_price"].iloc[0])
    final_inventory = float(all_period.loc[all_period["day_index"] == FORECAST_END_DAY, "inventory_remaining"].iloc[0])
    final_gap = float(all_period.loc[all_period["day_index"] == FORECAST_END_DAY, "supply_gap"].iloc[0])
    max_gap = float(forecast_period["supply_gap"].max())
    cutoff_price = float(all_period[all_period["阶段"] == "附件观测期"]["forecast_price"].iloc[-1])
    second_jump = max(post_observed_peak - cutoff_price, 0.0)
    risk_level = "低"
    if final_gap > 500 or second_jump > 15:
        risk_level = "高"
    elif final_gap > 150 or second_jump > 8:
        risk_level = "中"

    return {
        "scenario": scenario_key,
        "情景": SCENARIO_NAMES[scenario_key],
        **marker_prices,
        "外推期最高价": post_observed_peak,
        "外推期最低价": post_observed_min,
        "第180天价格": final_price,
        "外推期均价": float(forecast_period["forecast_price"].mean()),
        "第180天剩余商业库存": final_inventory,
        "第180天剩余供需缺口": final_gap,
        "外推期最大供需缺口": max_gap,
        "二次跳涨幅度": float(second_jump),
        "二次跳涨风险": risk_level,
    }
