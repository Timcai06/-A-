# 阶段 1 数据字典

本文档说明 `data/processed/brent_daily_clean.csv` 和 `data/processed/brent_event_window.csv` 的字段含义。两个文件字段一致，后者是前者按冲突窗口筛选后的子集。

| 字段 | 类型 | 含义 | 生成方式 |
|---|---|---|---|
| `trade_date` | date | 交易日期 | 由原始 `time` 字段解析 |
| `thscode` | text | 同花顺代码或合约标识 | 原始字段保留 |
| `pre_close` | float | 前一交易日收盘价 | 原始 `preClose` 字段重命名 |
| `open_price` | float | 开盘价 | 原始 `open` 字段重命名 |
| `high_price` | float | 盘中最高价 | 原始 `high` 字段重命名 |
| `low_price` | float | 盘中最低价 | 原始 `low` 字段重命名 |
| `close_price` | float | 收盘价 | 原始 `close` 字段重命名，后续拟合目标价 |
| `log_return` | float | 对数收益率 | `log(close_price / close_price.shift(1))` |
| `return_pct` | float | 普通收益率 | `close_price.pct_change()` |
| `volatility_7d` | float | 7 日滚动对数收益率标准差 | `log_return.rolling(7, min_periods=3).std()` |
| `volatility_14d` | float | 14 日滚动对数收益率标准差 | `log_return.rolling(14, min_periods=5).std()` |
| `volatility_30d` | float | 30 日滚动对数收益率标准差 | `log_return.rolling(30, min_periods=10).std()` |
| `is_event_window` | bool | 是否属于建模冲突窗口 | `2026-02-28 <= trade_date <= 2026-05-05` |

## 口径说明

- 后续模型校准建议以 `close_price` 作为真实价格目标。
- 附件 CSV 中冲突窗口最高收盘价为 `114.06`，最高盘中价为 `119.50`。
- 题面文字中提到的 `126 USD/barrel` 与附件 CSV 价格口径不完全一致，后续论文需单独说明。
- `pre_close` 第一行缺失来自原始文件的 `NA`。
- `log_return` 和 `return_pct` 第一行缺失是时间序列计算的自然结果。
