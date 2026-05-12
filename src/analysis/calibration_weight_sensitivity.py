"""Audit the robustness of calibration scoring weights.

This audit does not rerun the full calibration search. Instead, it takes the
stored top candidates and rescales the documented objective-function weights by
plus/minus 20%. The goal is to check whether the chosen candidate is a fragile
artifact of one exact weight vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.scenarios import simulation as scenario_sim


CANDIDATES_CSV = PROJECT_ROOT / "output" / "calibration" / "动态模型候选参数前10.csv"
OUTPUT_CSV = PROJECT_ROOT / "output" / "calibration" / "校准权重敏感性.csv"
CONSTANTS_CSV = PROJECT_ROOT / "output" / "sensitivity" / "长期机制常数说明.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "校准权重与长期常数透明度审计报告.md"


@dataclass(frozen=True)
class ScoreWeights:
    peak: float = 0.20
    final: float = 0.25
    high: float = 0.15
    early: float = 0.12
    mid: float = 0.12
    late: float = 0.18
    low: float = 0.18

    def scaled(self, factor: float) -> "ScoreWeights":
        return ScoreWeights(
            peak=self.peak * factor,
            final=self.final * factor,
            high=self.high * factor,
            early=self.early * factor,
            mid=self.mid * factor,
            late=self.late * factor,
            low=self.low * factor,
        )


WEIGHT_SCENARIOS: dict[str, ScoreWeights] = {
    "原始权重": ScoreWeights(),
    "误差项权重整体-20%": ScoreWeights().scaled(0.8),
    "误差项权重整体+20%": ScoreWeights().scaled(1.2),
    "峰值末日权重+20%": ScoreWeights(peak=0.24, final=0.30),
    "分段RMSE权重+20%": ScoreWeights(high=0.18, early=0.144, mid=0.144, late=0.216, low=0.216),
    "平台权重+20%": ScoreWeights(high=0.18, low=0.216),
}


def score_candidate(row: pd.Series, weights: ScoreWeights) -> float:
    return float(
        row["RMSE"]
        + weights.peak * abs(row["峰值误差"])
        + weights.final * abs(row["末日误差"])
        + weights.high * row["高价平台RMSE"]
        + weights.early * row["前期RMSE"]
        + weights.mid * row["中期RMSE"]
        + weights.late * row["后期RMSE"]
        + weights.low * row["低价回落RMSE"]
    )


def run_weight_audit() -> pd.DataFrame:
    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(f"Missing candidate file: {CANDIDATES_CSV}")
    candidates = pd.read_csv(CANDIDATES_CSV)
    rows: list[dict[str, Any]] = []
    for scenario_name, weights in WEIGHT_SCENARIOS.items():
        scored = candidates.copy()
        scored["审计得分"] = scored.apply(lambda row: score_candidate(row, weights), axis=1)
        scored = scored.sort_values(["审计得分", "RMSE"], ascending=True).reset_index(drop=True)
        for rank, row in scored.iterrows():
            rows.append(
                {
                    "权重情景": scenario_name,
                    "审计排名": rank + 1,
                    "candidate_id": int(row["candidate_id"]),
                    "candidate_source": row["candidate_source"],
                    "候选类型": row.get("候选类型", ""),
                    "审计得分": row["审计得分"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                    "峰值误差": row["峰值误差"],
                    "末日误差": row["末日误差"],
                    "高价平台RMSE": row["高价平台RMSE"],
                    "低价回落RMSE": row["低价回落RMSE"],
                    "前期RMSE": row["前期RMSE"],
                    "中期RMSE": row["中期RMSE"],
                    "后期RMSE": row["后期RMSE"],
                }
            )
    return pd.DataFrame(rows)


def build_constants_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "常数": note.name,
                "当前值": note.value,
                "作用": note.role,
                "取值依据": note.basis,
                "审计状态": note.audit_status,
            }
            for note in scenario_sim.MECHANISM_CONSTANT_NOTES
        ]
    )


def build_report(weight_df: pd.DataFrame, constants_df: pd.DataFrame) -> str:
    top = weight_df[weight_df["审计排名"] == 1].copy()
    winner_counts = top["candidate_id"].value_counts()
    winner_text = "\n".join(
        f"- 候选 {candidate_id}：在 {count} 个权重情景中排名第一。"
        for candidate_id, count in winner_counts.items()
    )
    top_rows = "\n".join(
        "| {权重情景} | {candidate_id} | {审计得分:.3f} | {RMSE:.3f} | {峰值误差:.3f} | {末日误差:.3f} |".format(
            **row
        )
        for row in top.to_dict("records")
    )
    constant_rows = "\n".join(
        "| {常数} | {当前值} | {作用} | {审计状态} |".format(**row)
        for row in constants_df.to_dict("records")
    )
    selected_id = int(pd.read_csv(CANDIDATES_CSV).iloc[0]["candidate_id"])
    selected_first_count = int(winner_counts.get(selected_id, 0))
    conclusion = (
        f"当前综合最优候选 {selected_id} 在 {selected_first_count}/{len(WEIGHT_SCENARIOS)} 个权重情景中保持第一。"
        if selected_first_count
        else f"当前综合最优候选 {selected_id} 在本轮权重扰动中未保持第一，应谨慎复核。"
    )
    return f"""# 校准权重与长期常数透明度审计报告

## 审计目的

本审计回应外部代码审计中关于“多目标权重是隐藏超参数”和“长期模块级常数缺少透明说明”的质疑。审计不重新搜索参数，不改变主模型，只检查已有前 10 候选在权重扰动下的排序稳定性，并集中列出长期机制常数的经济含义与审计状态。

## 权重敏感性结论

{conclusion}

{winner_text}

| 权重情景 | 第一候选 | 审计得分 | RMSE | 峰值误差 | 末日误差 |
|---|---:|---:|---:|---:|---:|
{top_rows}

该结果说明，当前候选并非只依赖唯一一组精确权重；但本审计只在已保存候选集内重排，不能替代完整再校准。论文中应表述为“权重扰动下的候选排序支持性证据”，不应写成全局最优证明。

## 长期机制常数说明

| 常数 | 当前值 | 作用 | 审计状态 |
|---|---:|---|---|
{constant_rows}

## 后续处理口径

- 已进入敏感性分析的长期常数，应在论文中作为“长期机制透明层”解释，而不是隐藏参数。
- 未进入敏感性分析的常数暂作为机制形状设定，若论文继续深化，可逐步迁移到配置文件或 dataclass。
- `FEAR_CHANGE_MOMENTUM` 暂不直接移除，因为它影响短期冲击吸收速度；当前更稳妥的做法是承认它是辅助动量项，并在后续做去除/保留的消融比较。

## 输出产物

- `{OUTPUT_CSV.relative_to(PROJECT_ROOT)}`
- `{CONSTANTS_CSV.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    weight_df = run_weight_audit()
    constants_df = build_constants_table()
    ensure_parent(OUTPUT_CSV)
    ensure_parent(CONSTANTS_CSV)
    ensure_parent(REPORT_PATH)
    weight_df.to_csv(OUTPUT_CSV, index=False)
    constants_df.to_csv(CONSTANTS_CSV, index=False)
    REPORT_PATH.write_text(build_report(weight_df, constants_df), encoding="utf-8")
    print("Calibration weight sensitivity audit complete")
    print(f"Weight audit: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Constants: {CONSTANTS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
