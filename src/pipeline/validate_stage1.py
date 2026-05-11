"""Automated validation for Stage 1 data-cleaning outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import yaml

from src.common.paths import PROJECT_ROOT
from src.pipeline import clean_data


CONFIG_PATH = PROJECT_ROOT / "config" / "base.yml"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def add_check(results: list[CheckResult], name: str, ok: bool, detail: str = "") -> None:
    results.append(CheckResult(name=name, status="PASS" if ok else "FAIL", detail=detail))


def load_stage1_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = config["paths"]
    raw = pd.read_csv(PROJECT_ROOT / paths["raw_brent_csv"], na_values=["NA", ""])
    full = pd.read_csv(PROJECT_ROOT / paths["processed_daily_csv"], parse_dates=["trade_date"])
    event = pd.read_csv(PROJECT_ROOT / paths["processed_event_window_csv"], parse_dates=["trade_date"])
    return raw, full, event


def find_ohlc_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    anomaly_mask = (
        (df["high_price"] < df["low_price"])
        | (df["high_price"] < df["open_price"])
        | (df["high_price"] < df["close_price"])
        | (df["low_price"] > df["open_price"])
        | (df["low_price"] > df["close_price"])
    )
    return df.loc[anomaly_mask].copy()


def validate_data(config: dict[str, Any], results: list[CheckResult]) -> dict[str, Any]:
    raw, full, event = load_stage1_frames(config)
    window = config["event_window"]
    start = pd.Timestamp(window["conflict_start"])
    end = pd.Timestamp(window["data_end"])

    add_check(results, "row_count_preserved", len(raw) == len(full), f"raw={len(raw)}, full={len(full)}")
    add_check(
        results,
        "event_row_count_matches_flag",
        len(event) == int(full["is_event_window"].sum()),
        f"event={len(event)}, flag_sum={int(full['is_event_window'].sum())}",
    )
    add_check(results, "date_monotonic_increasing", full["trade_date"].is_monotonic_increasing)
    add_check(results, "date_unique", full["trade_date"].is_unique, f"duplicates={full['trade_date'].duplicated().sum()}")
    add_check(
        results,
        "date_range_expected",
        str(full["trade_date"].min().date()) == "2017-09-01"
        and str(full["trade_date"].max().date()) == "2026-05-05",
        f"{full['trade_date'].min().date()} -> {full['trade_date'].max().date()}",
    )
    add_check(results, "event_window_within_config", event["trade_date"].between(start, end).all())
    add_check(
        results,
        "event_window_actual_trading_days",
        str(event["trade_date"].min().date()) == "2026-03-02"
        and str(event["trade_date"].max().date()) == "2026-05-05",
        f"{event['trade_date'].min().date()} -> {event['trade_date'].max().date()}",
    )

    for column in ["pre_close", "open_price", "high_price", "low_price", "close_price"]:
        add_check(results, f"{column}_numeric", pd.api.types.is_numeric_dtype(full[column]), str(full[column].dtype))

    add_check(results, "close_price_no_missing", full["close_price"].isna().sum() == 0)
    add_check(
        results,
        "pre_close_only_first_missing",
        full["pre_close"].isna().sum() == 1 and pd.isna(full.loc[0, "pre_close"]),
        f"missing={full['pre_close'].isna().sum()}",
    )
    add_check(
        results,
        "positive_ohlc_prices",
        (full[["open_price", "high_price", "low_price", "close_price"]] > 0).all().all(),
    )

    anomalies = find_ohlc_anomalies(full)
    add_check(
        results,
        "ohlc_anomalies_recorded_and_outside_event_window",
        len(anomalies) == 2 and not anomalies["is_event_window"].any(),
        f"anomalies={len(anomalies)}, in_event={int(anomalies['is_event_window'].sum()) if len(anomalies) else 0}",
    )

    expected_log = np.log(full["close_price"] / full["close_price"].shift(1))
    expected_pct = full["close_price"].pct_change()
    log_diff = (full["log_return"] - expected_log).abs().max(skipna=True)
    pct_diff = (full["return_pct"] - expected_pct).abs().max(skipna=True)
    add_check(results, "log_return_formula", log_diff < 1e-12, f"max_diff={log_diff}")
    add_check(results, "return_pct_formula", pct_diff < 1e-12, f"max_diff={pct_diff}")

    for window_size, min_periods, column in [
        (7, 3, "volatility_7d"),
        (14, 5, "volatility_14d"),
        (30, 10, "volatility_30d"),
    ]:
        expected = full["log_return"].rolling(window=window_size, min_periods=min_periods).std()
        diff = (full[column] - expected).abs().max(skipna=True)
        add_check(results, f"{column}_formula", diff < 1e-12, f"max_diff={diff}")

    add_check(results, "event_close_max_actual", abs(event["close_price"].max() - 114.06) < 1e-9)
    add_check(results, "event_high_max_actual", abs(event["high_price"].max() - 119.5) < 1e-9)
    add_check(results, "event_final_close_actual", abs(event["close_price"].iloc[-1] - 110.31) < 1e-9)

    return {
        "raw_rows": int(len(raw)),
        "processed_rows": int(len(full)),
        "event_rows": int(len(event)),
        "date_start": str(full["trade_date"].min().date()),
        "date_end": str(full["trade_date"].max().date()),
        "event_start": str(event["trade_date"].min().date()),
        "event_end": str(event["trade_date"].max().date()),
        "event_close_min": float(event["close_price"].min()),
        "event_close_max": float(event["close_price"].max()),
        "event_high_max": float(event["high_price"].max()),
        "event_final_close": float(event["close_price"].iloc[-1]),
        "ohlc_anomalies": int(len(anomalies)),
    }


def validate_figures(config: dict[str, Any], results: list[CheckResult]) -> list[dict[str, Any]]:
    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    figure_names = ["price_trend.png", "event_window_price.png", "return_volatility.png"]
    figure_metadata: list[dict[str, Any]] = []
    for name in figure_names:
        path = figures_dir / name
        exists = path.exists()
        add_check(results, f"figure_exists_{name}", exists, rel(path))
        if not exists:
            continue
        arr = mpimg.imread(path)
        pixel_std = float(np.nanstd(arr))
        nonblank = pixel_std > 0.001
        add_check(results, f"figure_nonblank_{name}", nonblank, f"std={pixel_std:.6f}")
        figure_metadata.append(
            {
                "path": rel(path),
                "bytes": path.stat().st_size,
                "shape": list(arr.shape),
                "pixel_std": pixel_std,
                "sha256_16": sha256_16(path),
            }
        )
    return figure_metadata


def capture_hashes(config: dict[str, Any]) -> dict[str, str]:
    paths = config["paths"]
    files = [
        PROJECT_ROOT / paths["processed_daily_csv"],
        PROJECT_ROOT / paths["processed_event_window_csv"],
        PROJECT_ROOT / paths["reports_dir"] / "阶段1_OHLC异常记录.csv",
    ]
    return {rel(path): sha256_16(path) for path in files if path.exists()}


def validate_reproducibility(config: dict[str, Any], results: list[CheckResult], rerun: bool) -> dict[str, Any]:
    if not rerun:
        return {"rerun": False, "checked": False}

    before = capture_hashes(config)
    clean_data.main()
    after = capture_hashes(config)
    for path, before_hash in before.items():
        after_hash = after.get(path)
        add_check(results, f"reproducible_{path}", before_hash == after_hash, f"before={before_hash}, after={after_hash}")
    return {"rerun": True, "checked": True, "before": before, "after": after}


def write_outputs(
    config: dict[str, Any],
    results: list[CheckResult],
    data_summary: dict[str, Any],
    figure_metadata: list[dict[str, Any]],
    reproducibility: dict[str, Any],
) -> None:
    reports_dir = PROJECT_ROOT / config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    result_rows = [result.__dict__ for result in results]
    results_csv = reports_dir / "阶段1_自动验收结果.csv"
    pd.DataFrame(result_rows).to_csv(results_csv, index=False)

    manifest = {
        "stage": "stage1_data_cleaning",
        "status": "PASS" if all(result.status == "PASS" for result in results) else "FAIL",
        "data_summary": data_summary,
        "figure_metadata": figure_metadata,
        "file_hashes": capture_hashes(config),
        "reproducibility": reproducibility,
    }
    manifest_path = reports_dir / "stage1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    pass_count = sum(1 for result in results if result.status == "PASS")
    fail_count = sum(1 for result in results if result.status == "FAIL")
    details = "\n".join(f"- {result.status}: `{result.name}` {result.detail}" for result in results)
    figures = "\n".join(
        f"- `{item['path']}`: shape={item['shape']}, std={item['pixel_std']:.6f}, sha256={item['sha256_16']}"
        for item in figure_metadata
    )
    report = f"""# 阶段 1 自动验收报告

## 结论

- 总检查数：{len(results)}
- 通过：{pass_count}
- 失败：{fail_count}
- 总体状态：{manifest["status"]}

## 数据摘要

- 全样本行数：{data_summary["processed_rows"]}
- 日期范围：{data_summary["date_start"]} 至 {data_summary["date_end"]}
- 冲突窗口行数：{data_summary["event_rows"]}
- 冲突窗口实际交易日：{data_summary["event_start"]} 至 {data_summary["event_end"]}
- 冲突窗口最高收盘价：{data_summary["event_close_max"]}
- 冲突窗口最高盘中价：{data_summary["event_high_max"]}
- OHLC 异常行数：{data_summary["ohlc_anomalies"]}

## 图表检查

{figures}

## 检查明细

{details}

## 机器可读文件

- `{rel(results_csv)}`
- `{rel(manifest_path)}`
"""
    (reports_dir / "stage1_validation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage 1 outputs.")
    parser.add_argument("--rerun", action="store_true", help="rerun Stage 1 pipeline and compare stable output hashes")
    args = parser.parse_args()

    config = load_config()
    results: list[CheckResult] = []
    data_summary = validate_data(config, results)
    figure_metadata = validate_figures(config, results)
    reproducibility = validate_reproducibility(config, results, args.rerun)
    write_outputs(config, results, data_summary, figure_metadata, reproducibility)

    failed = [result for result in results if result.status == "FAIL"]
    print(f"Stage 1 validation complete: {len(results) - len(failed)} passed, {len(failed)} failed")
    print("Report: output/reports/stage1_validation_report.md")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
