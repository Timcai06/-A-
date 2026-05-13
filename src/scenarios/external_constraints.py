"""Official external-data constraints for long-horizon scenario parameters.

The constraints here deliberately stay modest: JODI is a multi-country
reported sample rather than a complete world balance sheet, and EIA is a U.S.
series. They should tilt long-horizon scenario parameters toward externally
audited magnitudes, not overwrite the contest assumptions or the calibrated
short-term model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.models import dynamic_short_term as dynamic


JODI_SUMMARY_CSV = PROJECT_ROOT / "data" / "external" / "jodi" / "JODI多国外生约束_关键摘要.csv"
EIA_SUMMARY_CSV = PROJECT_ROOT / "data" / "external" / "eia" / "美国官方外生约束_关键摘要.csv"
OPEC_SUMMARY_CSV = PROJECT_ROOT / "data" / "external" / "opec" / "OPEC全球供需平衡摘要.csv"
CONSTRAINT_OUTPUT_CSV = PROJECT_ROOT / "output" / "scenarios" / "官方外生约束参数因子.csv"


@dataclass(frozen=True)
class ExternalConstraintFactors:
    route_capacity_multiplier: float = 1.0
    inventory_daily_cap_multiplier: float = 1.0
    demand_decline_multiplier: float = 1.0
    long_elasticity_multiplier: float = 1.0
    spr_release_multiplier: float = 1.0
    risk_weight_multiplier: float = 1.0
    uncertainty_floor_multiplier: float = 1.0
    evidence_note: str = "未接入外部数据，使用模型原参数。"


def _safe_metric(frame: pd.DataFrame, metric: str, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame.columns:
        return default
    row = frame[frame["指标"].astype(str) == metric]
    if row.empty:
        return default
    value = pd.to_numeric(row.iloc[0][column], errors="coerce")
    return default if pd.isna(value) else float(value)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_external_constraint_factors(write_output: bool = True) -> ExternalConstraintFactors:
    """Derive modest multipliers from EIA and JODI summary files."""
    jodi = _load_csv(JODI_SUMMARY_CSV)
    eia = _load_csv(EIA_SUMMARY_CSV)
    opec = _load_csv(OPEC_SUMMARY_CSV)
    if jodi.empty and eia.empty and opec.empty:
        factors = ExternalConstraintFactors()
        if write_output:
            write_constraint_table(factors)
        return factors

    production_change_kbd = _safe_metric(jodi, "多国原油及凝析油产量", "同口径较上年末变化")
    crude_stock_change_kbbl = _safe_metric(jodi, "多国原油及凝析油期末库存", "同口径较上年末变化")
    product_stock_change_kbbl = _safe_metric(jodi, "多国油品期末库存", "同口径较上年末变化")
    import_change_kbd = _safe_metric(jodi, "多国原油及凝析油进口", "同口径较上年末变化")
    export_change_kbd = _safe_metric(jodi, "多国原油及凝析油出口", "同口径较上年末变化")
    demand_change_kbd = _safe_metric(jodi, "多国油品总需求", "同口径较上年末变化")
    refinery_change_kbd = _safe_metric(jodi, "多国油品炼厂总产出", "同口径较上年末变化")

    eia_commercial_change_wan = _safe_metric(eia, "美国商业原油库存_不含SPR", "窗口变化_万桶口径")
    eia_spr_release_wan_day = abs(_safe_metric(eia, "美国SPR原油库存", "窗口日均变化_万桶每日期"))
    opec_demand_growth_wan = _safe_metric(opec, "OPEC全球需求年增量", "数值_万桶每日")
    opec_non_doc_growth_wan = _safe_metric(opec, "OPEC非DoC液体供给年增量", "数值_万桶每日")
    opec_doc_gap_growth_wan = _safe_metric(opec, "OPEC DoC原油需求差额年增量", "数值_万桶每日")
    opec_doc_gap_peak_above_avg_wan = _safe_metric(opec, "OPEC DoC原油需求差额峰值高于年均", "数值_万桶每日")

    # Positive stock changes mean there is audited buffer capacity, but the
    # adjustment is capped because JODI is a reported sample and EIA is U.S.-only.
    stock_change_wan = (crude_stock_change_kbbl + product_stock_change_kbbl) / 10.0
    inventory_multiplier = 1 + 0.55 * np.clip(stock_change_wan / 58_000.0, -0.08, 0.14)
    inventory_multiplier += 0.25 * np.clip(eia_commercial_change_wan / 58_000.0, -0.04, 0.06)
    inventory_multiplier = float(np.clip(inventory_multiplier, 0.94, 1.12))

    # Replacement logistics should be tightened when JODI trade flows contract.
    # Refinery output partly offsets that tightening because it indicates system
    # absorption capacity outside the single shipping channel.
    trade_stress_kbd = abs(min(import_change_kbd, 0.0)) + abs(min(export_change_kbd, 0.0))
    route_multiplier = 1 + 0.018 * np.clip(production_change_kbd / 1_000.0, -1.0, 1.0)
    route_multiplier += 0.010 * np.clip(refinery_change_kbd / 5_000.0, -1.0, 1.0)
    route_multiplier -= 0.040 * np.clip(trade_stress_kbd / 1_500.0, 0.0, 1.0)
    route_multiplier = float(np.clip(route_multiplier, 0.90, 1.04))

    # If reported demand is still rising before the shock, long-horizon demand
    # destruction should not be made too aggressive.
    demand_pressure = np.clip(max(demand_change_kbd, 0.0) / 5_000.0, 0.0, 1.0)
    opec_demand_pressure = np.clip(max(opec_demand_growth_wan - opec_non_doc_growth_wan, 0.0) / 120.0, 0.0, 1.0)
    demand_multiplier = float(np.clip(1 - 0.25 * demand_pressure, 0.90, 1.04))
    elasticity_multiplier = float(np.clip(1 - 0.14 * demand_pressure, 0.94, 1.03))
    demand_multiplier = float(np.clip(demand_multiplier * (1 - 0.035 * opec_demand_pressure), 0.88, 1.04))
    elasticity_multiplier = float(np.clip(elasticity_multiplier * (1 - 0.020 * opec_demand_pressure), 0.93, 1.03))

    # U.S. SPR release is not a global SPR proxy. Use it only as a gentle
    # credibility discount when it is extremely small relative to the contest
    # lower-bound release assumption.
    spr_multiplier = 0.96 + 0.04 * np.clip(eia_spr_release_wan_day / 200.0, 0.0, 1.0)
    spr_multiplier = float(np.clip(spr_multiplier, 0.94, 1.0))

    risk_multiplier = float(np.clip(1 + 0.055 * np.clip(trade_stress_kbd / 2_000.0, 0.0, 1.0), 1.0, 1.06))
    uncertainty_multiplier = float(np.clip(1 + 0.030 * np.clip(trade_stress_kbd / 2_000.0, 0.0, 1.0), 1.0, 1.04))
    opec_tail_pressure = np.clip((opec_doc_gap_growth_wan + opec_doc_gap_peak_above_avg_wan) / 180.0, 0.0, 1.0)
    risk_multiplier = float(np.clip(risk_multiplier * (1 + 0.020 * opec_tail_pressure), 1.0, 1.08))
    uncertainty_multiplier = float(np.clip(uncertainty_multiplier * (1 + 0.012 * opec_tail_pressure), 1.0, 1.05))

    factors = ExternalConstraintFactors(
        route_capacity_multiplier=route_multiplier,
        inventory_daily_cap_multiplier=inventory_multiplier,
        demand_decline_multiplier=demand_multiplier,
        long_elasticity_multiplier=elasticity_multiplier,
        spr_release_multiplier=spr_multiplier,
        risk_weight_multiplier=risk_multiplier,
        uncertainty_floor_multiplier=uncertainty_multiplier,
        evidence_note=(
            "基于 JODI 多国同口径产量、库存、进出口、需求和炼厂产出，"
            "叠加 EIA 美国 SPR 与商业库存周度变化，并引入 OPEC 全球供需平衡表"
            "约束长期需求基线和 DoC 需求差额尾部压力；仅作长期参数约束。"
        ),
    )
    if write_output:
        write_constraint_table(factors)
    return factors


def apply_external_constraints(
    scenario_key: str,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
    factors: ExternalConstraintFactors,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    """Apply external-data multipliers to scenario parameters."""
    scenario_intensity = {
        "optimistic": 0.55,
        "neutral": 1.00,
        "pessimistic": 1.20,
    }.get(scenario_key, 1.0)

    def blend(multiplier: float) -> float:
        return 1 + (multiplier - 1) * scenario_intensity

    constrained_assumptions = dynamic.PhysicalAssumptions(
        **{
            **asdict(assumptions),
            "spr_max_release": float(np.clip(assumptions.spr_max_release * blend(factors.spr_release_multiplier), 200, 700)),
            "route_max_capacity": float(np.clip(assumptions.route_max_capacity * blend(factors.route_capacity_multiplier), 120, 300)),
            "inventory_daily_cap": float(np.clip(assumptions.inventory_daily_cap * blend(factors.inventory_daily_cap_multiplier), 120, 520)),
            "observed_demand_decline": float(np.clip(assumptions.observed_demand_decline * blend(factors.demand_decline_multiplier), 250, 560)),
            "long_elasticity": float(np.clip(assumptions.long_elasticity * blend(factors.long_elasticity_multiplier), -0.36, -0.08)),
        }
    )
    constrained_behavior = dynamic.BehavioralParameters(
        **{
            **asdict(behavior),
            "risk_weight": float(behavior.risk_weight * blend(factors.risk_weight_multiplier)),
            "uncertainty_floor": float(np.clip(behavior.uncertainty_floor * blend(factors.uncertainty_floor_multiplier), 0.12, 0.42)),
        }
    )
    return constrained_assumptions, constrained_behavior


def write_constraint_table(factors: ExternalConstraintFactors) -> None:
    ensure_parent(CONSTRAINT_OUTPUT_CSV)
    rows = [
        {
            "参数因子": "route_capacity_multiplier",
            "中文含义": "绕道和替代贸易恢复能力乘数",
            "数值": factors.route_capacity_multiplier,
        },
        {
            "参数因子": "inventory_daily_cap_multiplier",
            "中文含义": "商业库存日缓冲上限乘数",
            "数值": factors.inventory_daily_cap_multiplier,
        },
        {
            "参数因子": "demand_decline_multiplier",
            "中文含义": "长期需求收缩幅度乘数",
            "数值": factors.demand_decline_multiplier,
        },
        {
            "参数因子": "long_elasticity_multiplier",
            "中文含义": "长期需求弹性绝对值乘数",
            "数值": factors.long_elasticity_multiplier,
        },
        {
            "参数因子": "spr_release_multiplier",
            "中文含义": "SPR 释放上限可信度乘数",
            "数值": factors.spr_release_multiplier,
        },
        {
            "参数因子": "risk_weight_multiplier",
            "中文含义": "贸易收缩风险权重乘数",
            "数值": factors.risk_weight_multiplier,
        },
        {
            "参数因子": "uncertainty_floor_multiplier",
            "中文含义": "长期不确定性强度乘数",
            "数值": factors.uncertainty_floor_multiplier,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["证据说明"] = factors.evidence_note
    frame.to_csv(CONSTRAINT_OUTPUT_CSV, index=False)
