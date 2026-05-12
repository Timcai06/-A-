#!/usr/bin/env Rscript

# R-side econometric audit for the oil-price modeling project.
# It is intentionally an auxiliary layer: Python remains the source of truth
# for the mechanism model, while R provides independent statistical checks.

suppressPackageStartupMessages({
  library(forecast)
  library(tseries)
  library(lmtest)
  library(sandwich)
  library(FinTS)
  library(rugarch)
  library(readr)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/audit/r_econometric_audit.R"
root <- normalizePath(file.path(dirname(script_path), "../.."))

calibrated_path <- file.path(root, "output", "calibration", "动态模型校准后路径.csv")
history_path <- file.path(root, "data", "processed", "布伦特原油期货主力合约价格数据_清洗后.csv")
output_dir <- file.path(root, "output", "statistics")
report_path <- file.path(root, "output", "reports", "R语言计量审计报告.md")
figure_path <- file.path(root, "figures", "R计量审计诊断.png")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(report_path), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(figure_path), recursive = TRUE, showWarnings = FALSE)

safe_p <- function(expr) {
  tryCatch(expr, error = function(e) NA_real_, warning = function(w) suppressWarnings(expr))
}

fmt <- function(x, digits = 4) {
  ifelse(is.na(x), "NA", format(round(x, digits), nsmall = digits))
}

rmse <- function(x) sqrt(mean(x^2, na.rm = TRUE))
mae <- function(x) mean(abs(x), na.rm = TRUE)
mape <- function(actual, predicted) mean(abs(predicted - actual) / abs(actual), na.rm = TRUE) * 100

model_df <- read_csv(calibrated_path, show_col_types = FALSE) %>%
  arrange(trade_date) %>%
  mutate(
    naive_forecast = lag(actual_price),
    model_error = simulated_price - actual_price,
    naive_error = naive_forecast - actual_price,
    model_abs_error = abs(model_error),
    naive_abs_error = abs(naive_error),
    model_squared_error = model_error^2,
    naive_squared_error = naive_error^2
  ) %>%
  filter(!is.na(naive_forecast))

history_df <- read_csv(history_path, show_col_types = FALSE) %>%
  arrange(trade_date) %>%
  filter(!is.na(close_price))

model_error <- model_df$model_error
naive_error <- model_df$naive_error

dm_sq <- tryCatch(
  forecast::dm.test(model_error, naive_error, alternative = "less", h = 1, power = 2),
  error = function(e) NULL
)
dm_abs <- tryCatch(
  forecast::dm.test(model_error, naive_error, alternative = "less", h = 1, power = 1),
  error = function(e) NULL
)

direction_actual <- sign(diff(model_df$actual_price))
direction_model <- sign(diff(model_df$simulated_price))
direction_valid <- direction_actual != 0
direction_hits <- sum(direction_actual[direction_valid] == direction_model[direction_valid])
direction_total <- sum(direction_valid)
direction_test <- binom.test(direction_hits, direction_total, p = 0.5, alternative = "greater")

calibration_lm <- lm(actual_price ~ simulated_price, data = model_df)
calibration_hac <- lmtest::coeftest(
  calibration_lm,
  vcov. = sandwich::NeweyWest(calibration_lm, lag = 3, prewhite = FALSE, adjust = TRUE)
)

arch_test <- FinTS::ArchTest(model_error, lags = 5)
lb_resid_5 <- Box.test(model_error, lag = 5, type = "Ljung-Box")
lb_resid_10 <- Box.test(model_error, lag = 10, type = "Ljung-Box")
lb_sq_5 <- Box.test(model_error^2, lag = 5, type = "Ljung-Box")
lb_sq_10 <- Box.test(model_error^2, lag = 10, type = "Ljung-Box")

garch_status <- "ok"
garch_alpha <- NA_real_
garch_beta <- NA_real_
garch_persistence <- NA_real_
garch_shape <- NA_real_
garch_fit <- tryCatch({
  spec <- rugarch::ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std"
  )
  rugarch::ugarchfit(spec, data = model_error, solver = "hybrid", solver.control = list(trace = 0))
}, error = function(e) {
  garch_status <<- paste("failed:", e$message)
  NULL
})

if (!is.null(garch_fit)) {
  garch_coef <- coef(garch_fit)
  garch_alpha <- unname(garch_coef["alpha1"])
  garch_beta <- unname(garch_coef["beta1"])
  garch_persistence <- garch_alpha + garch_beta
  garch_shape <- if ("shape" %in% names(garch_coef)) unname(garch_coef["shape"]) else NA_real_
}

log_price <- log(history_df$close_price)
log_return <- diff(log_price)
adf_price <- suppressWarnings(tseries::adf.test(history_df$close_price))
adf_log_price <- suppressWarnings(tseries::adf.test(log_price))
adf_return <- suppressWarnings(tseries::adf.test(log_return))

summary <- tibble::tibble(
  metric = c(
    "model_rmse",
    "naive_rmse",
    "model_mae",
    "naive_mae",
    "model_mape",
    "naive_mape",
    "rmse_improvement_pct",
    "dm_squared_statistic",
    "dm_squared_p_value",
    "dm_absolute_statistic",
    "dm_absolute_p_value",
    "direction_hit_rate_pct",
    "direction_p_value",
    "hac_intercept_estimate",
    "hac_intercept_p_value",
    "hac_slope_estimate",
    "hac_slope_p_value",
    "arch_lm_p_value",
    "garch_alpha1",
    "garch_beta1",
    "garch_persistence",
    "garch_shape"
  ),
  value = c(
    rmse(model_error),
    rmse(naive_error),
    mae(model_error),
    mae(naive_error),
    mape(model_df$actual_price, model_df$simulated_price),
    mape(model_df$actual_price, model_df$naive_forecast),
    (1 - rmse(model_error) / rmse(naive_error)) * 100,
    if (!is.null(dm_sq)) unname(dm_sq$statistic) else NA_real_,
    if (!is.null(dm_sq)) dm_sq$p.value else NA_real_,
    if (!is.null(dm_abs)) unname(dm_abs$statistic) else NA_real_,
    if (!is.null(dm_abs)) dm_abs$p.value else NA_real_,
    direction_hits / direction_total * 100,
    direction_test$p.value,
    calibration_hac[1, 1],
    calibration_hac[1, 4],
    calibration_hac[2, 1],
    calibration_hac[2, 4],
    arch_test$p.value,
    garch_alpha,
    garch_beta,
    garch_persistence,
    garch_shape
  )
)

diagnostics <- tibble::tibble(
  test = c(
    "Ljung-Box residual lag5",
    "Ljung-Box residual lag10",
    "Ljung-Box squared residual lag5",
    "Ljung-Box squared residual lag10",
    "ARCH-LM lag5",
    "GARCH(1,1)"
  ),
  statistic = c(
    unname(lb_resid_5$statistic),
    unname(lb_resid_10$statistic),
    unname(lb_sq_5$statistic),
    unname(lb_sq_10$statistic),
    unname(arch_test$statistic),
    NA_real_
  ),
  p_value = c(
    lb_resid_5$p.value,
    lb_resid_10$p.value,
    lb_sq_5$p.value,
    lb_sq_10$p.value,
    arch_test$p.value,
    NA_real_
  ),
  note = c(
    "短期模型残差线性自相关检验",
    "短期模型残差线性自相关检验",
    "平方残差波动聚集检验",
    "平方残差波动聚集检验",
    "条件异方差检验",
    garch_status
  )
)

historical_tests <- tibble::tibble(
  series = c("close_price_level", "log_close_price", "log_return"),
  adf_statistic = c(unname(adf_price$statistic), unname(adf_log_price$statistic), unname(adf_return$statistic)),
  p_value = c(adf_price$p.value, adf_log_price$p.value, adf_return$p.value),
  conclusion = c(
    "价格水平通常按非平稳序列处理",
    "对数价格通常按非平稳序列处理",
    "收益率序列可作为更合适的统计建模对象"
  )
)

write_csv(summary, file.path(output_dir, "R计量审计摘要.csv"))
write_csv(diagnostics, file.path(output_dir, "R残差计量诊断.csv"))
write_csv(historical_tests, file.path(output_dir, "R历史序列平稳性检验.csv"))

png(figure_path, width = 1800, height = 900, res = 180)
plot_df <- model_df %>%
  mutate(loss_diff = model_squared_error - naive_squared_error)
op <- par(mfrow = c(1, 3), mar = c(4, 4, 3, 1), family = "sans")
plot(
  as.Date(plot_df$trade_date), plot_df$model_error,
  type = "l", col = "#2563eb", lwd = 2,
  xlab = "", ylab = "USD/barrel", main = "Model residuals"
)
abline(h = 0, col = "#1f2937", lwd = 1)
grid(col = "#e5e7eb")
hist(
  plot_df$model_error, breaks = 12,
  col = "#2563eb", border = "white",
  xlab = "USD/barrel", main = "Residual distribution"
)
abline(v = 0, col = "#1f2937", lwd = 1)
plot(
  as.Date(plot_df$trade_date), plot_df$loss_diff,
  type = "l", col = "#dc2626", lwd = 2,
  xlab = "", ylab = "Negative favors model",
  main = "Squared loss difference vs naive"
)
abline(h = 0, col = "#1f2937", lwd = 1)
grid(col = "#e5e7eb")
par(op)
dev.off()

report <- paste0(
  "# R语言计量审计报告\n\n",
  "## 审计定位\n\n",
  "本报告由 R 语言独立生成，不改变 Python 主模型。R 的作用是补充计量经济学检验，",
  "用于增强论文中“模型检验”和“稳健性分析”的可信度。\n\n",
  "## 核心结果\n\n",
  "- 机制模型 RMSE 为 ", fmt(summary$value[summary$metric == "model_rmse"], 2),
  "，朴素上一日基准 RMSE 为 ", fmt(summary$value[summary$metric == "naive_rmse"], 2),
  "，RMSE 改善率为 ", fmt(summary$value[summary$metric == "rmse_improvement_pct"], 2), "%。\n",
  "- R 版 DM 平方损失检验 p 值为 ", fmt(summary$value[summary$metric == "dm_squared_p_value"], 4),
  "；绝对损失检验 p 值为 ", fmt(summary$value[summary$metric == "dm_absolute_p_value"], 4), "。\n",
  "- 方向命中率为 ", fmt(summary$value[summary$metric == "direction_hit_rate_pct"], 2),
  "%，二项检验 p 值为 ", fmt(summary$value[summary$metric == "direction_p_value"], 4), "。\n",
  "- Newey-West 稳健校准回归中，斜率估计为 ",
  fmt(summary$value[summary$metric == "hac_slope_estimate"], 4),
  "，用于检查拟合值与真实价格之间的线性校准关系。\n",
  "- ARCH-LM p 值为 ", fmt(summary$value[summary$metric == "arch_lm_p_value"], 4),
  "；GARCH 持续性 alpha+beta 为 ", fmt(summary$value[summary$metric == "garch_persistence"], 4),
  "。该结果只作为波动聚集提示，不替代机制模型。\n\n",
  "## 论文可用解释\n\n",
  "R 语言审计的价值不在于另建一个主模型，而在于用成熟计量工具验证本文模型是否真正优于朴素基准、",
  "残差是否存在明显自相关、金融波动聚集是否需要被承认。由于冲突窗口样本较短，",
  "本文将 R 输出作为支持性证据，而不把任何单一 p 值写成绝对证明。\n\n",
  "## 输出文件\n\n",
  "- `output/statistics/R计量审计摘要.csv`\n",
  "- `output/statistics/R残差计量诊断.csv`\n",
  "- `output/statistics/R历史序列平稳性检验.csv`\n",
  "- `figures/R计量审计诊断.png`\n"
)

writeLines(report, report_path, useBytes = TRUE)

cat("R econometric audit complete\n")
cat(file.path(output_dir, "R计量审计摘要.csv"), "\n")
cat(file.path(output_dir, "R残差计量诊断.csv"), "\n")
cat(file.path(output_dir, "R历史序列平稳性检验.csv"), "\n")
cat(report_path, "\n")
cat(figure_path, "\n")
