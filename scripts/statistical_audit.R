#!/usr/bin/env Rscript

# Optional R-side statistical audit for the short-term model.
# This mirrors Stage 13's core checks and is kept as a reproducible R entry
# point once R is installed on the machine.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/statistical_audit.R"
root <- normalizePath(file.path(dirname(script_path), ".."))
input_path <- file.path(root, "output", "calibration", "动态模型校准后路径.csv")
output_dir <- file.path(root, "output", "statistics")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

dm_test <- function(model_loss, baseline_loss) {
  d <- model_loss - baseline_loss
  n <- length(d)
  mean_d <- mean(d)
  sd_d <- sd(d)
  statistic <- mean_d / (sd_d / sqrt(n))
  p_value <- pt(statistic, df = n - 1)
  data.frame(mean_diff = mean_d, statistic = statistic, p_value = p_value)
}

df <- read.csv(input_path, stringsAsFactors = FALSE)
df$naive_forecast <- c(NA, head(df$actual_price, -1))
eval_df <- df[complete.cases(df[, c("actual_price", "simulated_price", "naive_forecast")]), ]

model_error <- eval_df$simulated_price - eval_df$actual_price
naive_error <- eval_df$naive_forecast - eval_df$actual_price

rmse <- function(x) sqrt(mean(x^2))
mae <- function(x) mean(abs(x))
mape <- function(actual, predicted) mean(abs(predicted - actual) / abs(actual)) * 100

dm_sq <- dm_test(model_error^2, naive_error^2)
dm_abs <- dm_test(abs(model_error), abs(naive_error))

actual_direction <- sign(diff(eval_df$actual_price))
model_direction <- sign(diff(eval_df$simulated_price))
valid <- actual_direction != 0
direction_hits <- sum(actual_direction[valid] == model_direction[valid])
direction_total <- sum(valid)
direction_test <- binom.test(direction_hits, direction_total, p = 0.5, alternative = "greater")

summary <- data.frame(
  metric = c(
    "model_rmse",
    "naive_rmse",
    "model_mae",
    "naive_mae",
    "model_mape",
    "naive_mape",
    "dm_squared_loss_mean_diff",
    "dm_squared_loss_p_value",
    "dm_absolute_loss_mean_diff",
    "dm_absolute_loss_p_value",
    "direction_hit_rate",
    "direction_p_value"
  ),
  value = c(
    rmse(model_error),
    rmse(naive_error),
    mae(model_error),
    mae(naive_error),
    mape(eval_df$actual_price, eval_df$simulated_price),
    mape(eval_df$actual_price, eval_df$naive_forecast),
    dm_sq$mean_diff,
    dm_sq$p_value,
    dm_abs$mean_diff,
    dm_abs$p_value,
    direction_hits / direction_total * 100,
    direction_test$p.value
  )
)

write.csv(summary, file.path(output_dir, "R短期统计审计摘要.csv"), row.names = FALSE)
cat("R statistical audit complete\n")
cat(file.path(output_dir, "R短期统计审计摘要.csv"), "\n")
