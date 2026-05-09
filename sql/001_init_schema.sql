-- 数学建模 A 题 PostgreSQL 初始化结构
-- 当前文件仅作为阶段 0.6 设计产物，暂不自动执行。

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS processed;
CREATE SCHEMA IF NOT EXISTS model;
CREATE SCHEMA IF NOT EXISTS reporting;

CREATE TABLE IF NOT EXISTS raw.brent_daily_prices (
    id BIGSERIAL PRIMARY KEY,
    trade_date DATE NOT NULL UNIQUE,
    thscode TEXT,
    pre_close NUMERIC,
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC NOT NULL,
    source_file TEXT NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS processed.brent_daily_features (
    trade_date DATE PRIMARY KEY REFERENCES raw.brent_daily_prices(trade_date),
    close_price NUMERIC NOT NULL,
    log_return NUMERIC,
    return_pct NUMERIC,
    volatility_7d NUMERIC,
    volatility_14d NUMERIC,
    volatility_30d NUMERIC,
    is_event_window BOOLEAN NOT NULL DEFAULT false,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model.parameter_sets (
    parameter_set_id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    stage TEXT NOT NULL,
    description TEXT,
    source_type TEXT NOT NULL,
    parameters JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model.simulation_runs (
    run_id BIGSERIAL PRIMARY KEY,
    run_name TEXT NOT NULL,
    stage TEXT NOT NULL,
    model_name TEXT NOT NULL,
    parameter_set_id BIGINT REFERENCES model.parameter_sets(parameter_set_id),
    input_range_start DATE,
    input_range_end DATE,
    metrics JSONB,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model.price_paths (
    run_id BIGINT NOT NULL REFERENCES model.simulation_runs(run_id) ON DELETE CASCADE,
    day_index INTEGER NOT NULL,
    trade_date DATE,
    actual_price NUMERIC,
    simulated_price NUMERIC,
    effective_supply NUMERIC,
    effective_demand NUMERIC,
    inventory_level NUMERIC,
    spr_release NUMERIC,
    route_capacity NUMERIC,
    fear_factor NUMERIC,
    PRIMARY KEY (run_id, day_index)
);

CREATE TABLE IF NOT EXISTS model.sensitivity_results (
    sensitivity_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT REFERENCES model.simulation_runs(run_id) ON DELETE CASCADE,
    parameter_name TEXT NOT NULL,
    parameter_value NUMERIC NOT NULL,
    peak_price NUMERIC,
    final_price NUMERIC,
    mean_price NUMERIC,
    inventory_depletion_day INTEGER,
    metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reporting.figure_registry (
    figure_id TEXT PRIMARY KEY,
    figure_path TEXT NOT NULL,
    paper_section TEXT,
    supported_claim TEXT,
    source_table TEXT,
    source_file TEXT,
    status TEXT NOT NULL DEFAULT 'planned',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_raw_brent_trade_date
    ON raw.brent_daily_prices(trade_date);

CREATE INDEX IF NOT EXISTS idx_processed_event_window
    ON processed.brent_daily_features(is_event_window);

CREATE INDEX IF NOT EXISTS idx_price_paths_run_id
    ON model.price_paths(run_id);
