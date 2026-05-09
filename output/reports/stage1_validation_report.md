# 阶段 1 自动验收报告

## 结论

- 总检查数：33
- 通过：33
- 失败：0
- 总体状态：PASS

## 数据摘要

- 全样本行数：2237
- 日期范围：2017-09-01 至 2026-05-05
- 冲突窗口行数：46
- 冲突窗口实际交易日：2026-03-02 至 2026-05-05
- 冲突窗口最高收盘价：114.06
- 冲突窗口最高盘中价：119.5
- OHLC 异常行数：2

## 图表检查

- `figures/price_trend.png`: shape=[990, 1980, 4], std=0.093178, sha256=3b38fb00325b4e51
- `figures/event_window_price.png`: shape=[990, 1980, 4], std=0.088400, sha256=939f4060e8c8cfd7
- `figures/return_volatility.png`: shape=[1350, 1980, 4], std=0.149481, sha256=b1b232f380395ed1

## 检查明细

- PASS: `row_count_preserved` raw=2237, full=2237
- PASS: `event_row_count_matches_flag` event=46, flag_sum=46
- PASS: `date_monotonic_increasing` 
- PASS: `date_unique` duplicates=0
- PASS: `date_range_expected` 2017-09-01 -> 2026-05-05
- PASS: `event_window_within_config` 
- PASS: `event_window_actual_trading_days` 2026-03-02 -> 2026-05-05
- PASS: `pre_close_numeric` float64
- PASS: `open_price_numeric` float64
- PASS: `high_price_numeric` float64
- PASS: `low_price_numeric` float64
- PASS: `close_price_numeric` float64
- PASS: `close_price_no_missing` 
- PASS: `pre_close_only_first_missing` missing=1
- PASS: `positive_ohlc_prices` 
- PASS: `ohlc_anomalies_recorded_and_outside_event_window` anomalies=2, in_event=0
- PASS: `log_return_formula` max_diff=9.985502008591496e-17
- PASS: `return_pct_formula` max_diff=9.985502008591496e-17
- PASS: `volatility_7d_formula` max_diff=1.3548190347378863e-15
- PASS: `volatility_14d_formula` max_diff=2.8449465006019636e-16
- PASS: `volatility_30d_formula` max_diff=3.191891195797325e-16
- PASS: `event_close_max_actual` 
- PASS: `event_high_max_actual` 
- PASS: `event_final_close_actual` 
- PASS: `figure_exists_price_trend.png` figures/price_trend.png
- PASS: `figure_nonblank_price_trend.png` std=0.093178
- PASS: `figure_exists_event_window_price.png` figures/event_window_price.png
- PASS: `figure_nonblank_event_window_price.png` std=0.088400
- PASS: `figure_exists_return_volatility.png` figures/return_volatility.png
- PASS: `figure_nonblank_return_volatility.png` std=0.149481
- PASS: `reproducible_data/processed/brent_daily_clean.csv` before=d50d5ac97c39d6e7, after=d50d5ac97c39d6e7
- PASS: `reproducible_data/processed/brent_event_window.csv` before=6ff26ad44699bee5, after=6ff26ad44699bee5
- PASS: `reproducible_output/reports/stage1_ohlc_anomalies.csv` before=ea02e80a2ab65a2f, after=ea02e80a2ab65a2f

## 机器可读文件

- `output/reports/stage1_validation_results.csv`
- `output/reports/stage1_manifest.json`
