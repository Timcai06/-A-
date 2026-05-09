# 短期模型展示台

本目录用于展示阶段 1-4 的短期模型成果。当前实现采用 `Streamlit + Plotly + YAML`，重点展示：

- 实际油价与短期动态模型拟合曲线
- 每个交易日的模型误差
- 前期、中期、后期、高价平台、低价回落分段误差
- 供需缺口、封锁风险、不确定性、恐慌、缓冲确认、预期修复等机制贡献
- 阶段 4 候选参数对比

运行方式：

```bash
source scripts/project_env.sh
streamlit run dashboard/streamlit_app.py
```

展示配置：

```text
config/dashboard.yml
```

后续如果要做 `React + ECharts` 版本，可以复用 `config/dashboard.yml` 的路径和分段配置，并读取同一批 CSV 输出。
