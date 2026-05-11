"""Stage 8 factor selection matrix for the final integrated model."""

from __future__ import annotations

import pandas as pd

from src.common.paths import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "output" / "reports"
FACTOR_CSV = OUTPUT_DIR / "综合主模型因素覆盖矩阵.csv"
FACTOR_REPORT = OUTPUT_DIR / "综合主模型因素覆盖矩阵.md"


FACTORS = [
    ("供应中断量", "赛题明确要求", "物理供需层", "进入主模型", 5, 5, 5, 1, "决定基础供需缺口"),
    ("短期需求弹性", "赛题明确要求", "需求函数", "进入主模型", 5, 5, 5, 1, "解释低弹性理论高价"),
    ("SPR释放", "赛题明确要求", "供给补充", "进入主模型", 5, 5, 5, 2, "消融后误差显著变差"),
    ("商业库存", "赛题明确要求", "库存缓冲", "进入主模型", 4, 4, 4, 2, "影响短期缓冲和长期耗尽风险"),
    ("绕道运输", "赛题明确要求", "运输恢复", "进入主模型", 4, 4, 4, 2, "影响恢复路径和长期缺口"),
    ("恐慌情绪", "赛题明确要求", "风险溢价", "进入主模型", 4, 3, 3, 3, "解释冲突初期情绪放大"),
    ("需求收缩", "赛题隐含机制", "需求回落", "进入主模型", 4, 4, 4, 2, "消融后低价回落段恶化"),
    ("地缘风险溢价", "现实扩展因素", "价格形成层", "进入主模型", 5, 3, 5, 3, "解释高位风险重估"),
    ("不确定性平台", "现实扩展因素", "价格形成层", "进入主模型", 5, 3, 5, 3, "敏感性排序最高"),
    ("缓冲确认折价", "现实扩展因素", "价格形成层", "进入主模型", 4, 3, 4, 3, "解释缓冲被确认后的回落"),
    ("预期修复折价", "现实扩展因素", "价格形成层", "进入主模型", 4, 3, 5, 3, "消融后后期误差恶化"),
    ("OPEC+剩余产能", "现实扩展因素", "长期供给", "进入长期情景", 4, 2, 3, 2, "主要影响60-180天供给恢复"),
    ("非OPEC供给响应", "现实扩展因素", "长期供给", "进入长期情景", 3, 2, 2, 2, "短期反应慢，适合情景项"),
    ("油轮保险费和航运费率", "现实扩展因素", "风险解释", "暂不单独参数化", 4, 1, 2, 4, "缺少附件内直接观测"),
    ("美元指数和利率", "现实扩展因素", "宏观背景", "进入改进方向", 3, 1, 2, 3, "非赛题主线，需额外数据"),
    ("期货期限结构", "现实扩展因素", "市场结构", "进入改进方向", 4, 1, 3, 3, "需期限曲线数据"),
    ("投机资金和套保需求", "现实扩展因素", "风险解释", "暂不单独参数化", 3, 1, 2, 4, "识别难度高"),
    ("冲突升级概率", "现实扩展因素", "尾部风险", "进入悲观情景", 5, 2, 3, 3, "适合情景而非短期拟合参数"),
]


def build_factor_frame() -> pd.DataFrame:
    columns = [
        "因素",
        "来源",
        "当前处理方式",
        "纳入决策",
        "解释重要性",
        "附件数据可得性",
        "边际贡献预期",
        "过拟合风险",
        "备注",
    ]
    df = pd.DataFrame(FACTORS, columns=columns)
    df["综合纳入分"] = (
        df["解释重要性"] * 0.35
        + df["附件数据可得性"] * 0.25
        + df["边际贡献预期"] * 0.25
        - df["过拟合风险"] * 0.15
    ).round(2)
    decision_order = {
        "进入主模型": 1,
        "进入长期情景": 2,
        "进入悲观情景": 3,
        "进入改进方向": 4,
        "暂不单独参数化": 5,
    }
    df["决策排序"] = df["纳入决策"].map(decision_order).fillna(99)
    return (
        df.sort_values(["决策排序", "综合纳入分"], ascending=[True, False])
        .drop(columns=["决策排序"])
        .reset_index(drop=True)
    )


def write_report(df: pd.DataFrame) -> None:
    report_lines = [
        "# 综合主模型因素覆盖矩阵",
        "",
        "本报告由 `src/analysis/factor_selection_stage8.py` 生成，用于约束后续模型优化：不是把所有现实因素都塞进主模型，而是按解释重要性、附件数据可得性、边际贡献预期和过拟合风险进行分层。",
        "",
        "## 因素评分表",
        "",
        df.to_markdown(index=False),
        "",
        "## 当前建议",
        "",
        "- 赛题明确因素和已经通过消融/敏感性检验证明有效的价格形成因素，继续保留在综合机制递推模型中。",
        "- OPEC+ 剩余产能、非 OPEC 供给响应和冲突升级概率更适合进入 60-180 天情景，而不是短期拟合参数。",
        "- 油轮保险费、美元指数、期货期限结构、投机资金等因素可以写入论文的局限性和改进方向；若后续要进入模型，必须先获得真实外生数据，不能编造。",
        "- 下一轮数值优化应优先强化现有主模型的检验和解释，而不是继续增加自由参数。",
        "",
    ]
    FACTOR_REPORT.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_factor_frame()
    df.to_csv(FACTOR_CSV, index=False, encoding="utf-8-sig")
    write_report(df)
    print(f"Wrote {FACTOR_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {FACTOR_REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
