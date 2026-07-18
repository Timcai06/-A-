<div align="center">

# 霍尔木兹海峡封锁对国际原油价格的影响

**用机制模型解释短期价格平台，用状态转移与蒙特卡洛刻画长期尾部风险**

[![Python](https://img.shields.io/badge/Python-Scientific%20Stack-3776AB?logo=python&logoColor=white)](environment.yml)
[![R](https://img.shields.io/badge/R-Robustness%20Audit-276DC3?logo=r&logoColor=white)](src)
[![Model](https://img.shields.io/badge/Model-Mechanism%20%2B%20Monte%20Carlo-8A2BE2)](docs/01_建模方案.md)
[![Paper](https://img.shields.io/badge/Paper-XeLaTeX-008080?logo=latex&logoColor=white)](paper/总论文.tex)
[![Reproducible](https://img.shields.io/badge/Workflow-Reproducible-2E8B57)](scripts/build/build_final_paper.sh)

浙江工商大学 2026 年大学生数学建模竞赛 · A 题

</div>

![短期机制拟合与中长期三情景预测总览](paper/figures/短期拟合与长期预测总览.png)

## 研究问题

当霍尔木兹海峡封锁造成巨大供应缺口时，为什么真实油价没有冲向线性需求反事实给出的 278—337 美元/桶，而是在 110—120 美元/桶附近形成平台？

本项目从附件真实价格出发，将问题拆为两个时间尺度：先用 **0—60 天短期冲击模型**解释价格平台及其形成机制，再用 **60—180 天中长期油价调节模型**评估库存、SPR、替代供给、风险预期与信心恢复共同作用下的情景路径和尾部风险。常弹性需求模型只作为机械上界，不作为现实预测。

## 核心结论

| 模块 | 关键结果 |
|---|---|
| 短期冲击模型 | RMSE **3.38 美元/桶**，MAE **2.76 美元/桶**，MAPE **2.76%** |
| 阶段 Ridge 增强 | RMSE 进一步降至 **3.18 美元/桶**，仅作为滞后残差修正，不替代机制主模型 |
| 中长期情景 | 第 180 天中性情景约 **94.15 美元/桶**，悲观情景约 **110.16 美元/桶** |
| 蒙特卡洛 | 第 180 天价格中位数约 **96.69 美元/桶** |
| 尾部风险 | 外推期最高价突破 120 美元/桶的条件概率约 **16.3%** |

模型优于朴素上一日基准与滚动 ARIMA 基准，并通过 DM 检验、残差诊断、方向命中检验、历史滚动验证、参数扰动和机制消融进行多层审计。

## 建模路线

![论文总体技术路线](paper/figures/论文总体技术路线图.png)

1. **真实数据与反事实基准**：统一附件日期与价格口径，用线性需求和常弹性需求估计无缓冲机制时的价格上界。
2. **短期动态递推**：显式描述 SPR、商业库存、绕道运输、恐慌溢价和预期修复，解释 0—60 天价格平台。
3. **长期状态转移**：将库存消耗、替代供给、政策缓冲、信心恢复和过剩供给回归写入 60—180 天递推。
4. **风险量化**：基于 2017—2025 历史波动、OPEC 供需基线和 OVX 风险约束生成蒙特卡洛路径。
5. **稳健性审计**：联合 Python 与 R 完成基准对比、消融实验、统计检验、敏感性和历史窗口极端性检查。

## 关键图表

<table>
  <tr>
    <td width="50%"><img src="paper/figures/短期模型拟合效果.png" alt="短期模型拟合效果"></td>
    <td width="50%"><img src="paper/figures/短期模型基准对比.png" alt="短期模型基准对比"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>短期拟合：</b>机制模型跟踪冲突窗口内的价格平台</sub></td>
    <td align="center"><sub><b>基准比较：</b>与朴素预测及滚动 ARIMA 同口径评估</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="paper/figures/蒙特卡洛情景树高级组合图.png" alt="蒙特卡洛情景树与尾部概率"></td>
    <td width="50%"><img src="paper/figures/长期敏感性扰动热力图.png" alt="长期敏感性扰动热力图"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>概率预测：</b>路径云、分位数与阈值突破概率一体化表达</sub></td>
    <td align="center"><sub><b>敏感性：</b>识别长期价格与尾部风险的关键驱动参数</sub></td>
  </tr>
</table>

## 数据与证据边界

- **附件真实数据**：确定冲突窗口、短期拟合目标与预测起点；
- **EIA / JODI / OPEC**：约束库存、产量、供需平衡和替代供给参数；
- **OVX**：刻画市场隐含波动率，并进入长期风险权重与不确定性强度；
- **GPR 与历史 Brent 窗口**：用于滞后审计和极端性比较，不把相关性直接解释为因果；
- **期货期限结构与油轮保险费**：因缺少可复现的多到期合约或授权运输风险数据，未强行数值化。

这种边界刻意区分了“已进入模型的外生约束”“只用于审计的证据”和“暂不具备复现条件的数据”。

## 一键复现

```bash
source scripts/project_env.sh
./scripts/build/build_final_paper.sh
```

正式构建需要 `xelatex -shell-escape` 与 Pygments，以渲染论文附录中的代码。运行后生成：

- `output/final/A题_霍尔木兹海峡封锁对国际原油价格影响_论文.pdf`
- `output/final/A题_霍尔木兹海峡封锁对国际原油价格影响_论文.docx`

## 阅读入口

| 入口 | 内容 |
|---|---|
| [项目总览](docs/00_项目总览.md) | 当前状态、核心结果与交付边界 |
| [建模方案](docs/01_建模方案.md) | 短期/长期模型结构、因素纳入和公式口径 |
| [工程复现](docs/02_工程复现.md) | 环境、命令与计算流程 |
| [交付物与数据索引](docs/03_交付物与数据索引.md) | 关键 CSV、图表、报告和论文位置 |
| [决策与质疑防御](docs/04_决策与质疑防御.md) | 关键假设、建模选择与评委质疑回应 |
| [总论文 LaTeX](paper/总论文.tex) | 正式论文主控文件 |

## 仓库结构

| 路径 | 内容 |
|---|---|
| [`data/`](data) | 附件、清洗数据、外部审计数据和参数来源 |
| [`src/`](src) | 数据清洗、校准、建模、检验和可视化代码 |
| [`paper/`](paper) | 模块化论文源码、参考文献和正式图表 |
| [`output/reports/`](output/reports) | 阶段报告与模型证据 |
| [`dashboard/`](dashboard) | 可选 Streamlit 交互展示 |

> 最终交付只有一份总论文；历史素材稿不属于当前主线。所有关键结论均应能追溯到数据、脚本或审计结果。
