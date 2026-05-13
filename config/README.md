# config 目录说明

本目录保存模型和展示系统的配置文件。

| 文件 | 作用 |
|---|---|
| `base.yml` | 基础数据路径和主模型默认配置 |
| `calibration_search.yml` | 短期模型参数搜索与校准范围 |
| `scenarios.yml` | 长期乐观、中性、悲观三情景参数 |
| `experiments.yml` | 模型实验管理层配置，统一汇总短期、长期、反事实和质量闸门指标 |
| `dashboard.yml` | Streamlit 展示台配置 |
| `database.yml` | PostgreSQL 方案配置草案，当前不是主线 |
| `变量定义表.csv` | 变量中文定义和口径说明 |

修改配置后，应重新运行对应阶段脚本并更新 `output/reports/` 中的证据报告。
