#!/usr/bin/env Rscript

# Academic ggplot2 figures for the final paper.
# Python remains the modeling source of truth; this R script only improves
# visual presentation and statistical reporting artifacts.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/audit/r_academic_figures.R"
root <- normalizePath(file.path(dirname(script_path), "../.."))

fig_dir <- file.path(root, "figures")
report_dir <- file.path(root, "output", "reports")
stat_dir <- file.path(root, "output", "statistics")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(stat_dir, recursive = TRUE, showWarnings = FALSE)

chinese_font <- "STHeiti"

calibrated_path <- file.path(root, "output", "calibration", "动态模型校准后路径.csv")
history_baseline_path <- file.path(root, "output", "history", "历史基准误差分布.csv")
state_quantile_path <- file.path(root, "output", "scenarios", "长期状态转移路径分位数.csv")
scenario_path <- file.path(root, "output", "scenarios", "三情景预测结果.csv")

short_error_fig <- file.path(fig_dir, "R短期误差学术诊断.png")
state_fan_fig <- file.path(fig_dir, "R长期状态转移扇形图.png")
history_baseline_fig <- file.path(fig_dir, "R历史基准误差分布图.png")
summary_csv <- file.path(stat_dir, "R学术图表增强摘要.csv")
report_path <- file.path(report_dir, "R学术图表增强报告.md")

fmt <- function(x, digits = 2) {
  format(round(as.numeric(x), digits), nsmall = digits)
}

date_label_cn <- function(x) {
  format(x, "%m月%d日")
}

paper_theme <- function(base_size = 12) {
  theme_minimal(base_size = base_size, base_family = chinese_font) +
    theme(
      text = element_text(family = chinese_font),
      plot.title = element_text(family = chinese_font, face = "bold", colour = "#111827", size = base_size + 2),
      plot.subtitle = element_text(family = chinese_font, colour = "#4b5563", size = base_size - 1),
      axis.title = element_text(family = chinese_font, colour = "#374151"),
      axis.text = element_text(family = chinese_font, colour = "#374151"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#e5e7eb", linewidth = 0.35),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(family = chinese_font),
      plot.caption = element_text(family = chinese_font, colour = "#6b7280", size = base_size - 3)
    )
}

save_plot <- function(plot, path, width = 9.5, height = 5.6) {
  ggsave(
    path,
    plot = plot,
    width = width,
    height = height,
    dpi = 260,
    bg = "white",
    device = function(filename, width, height, ...) {
      grDevices::png(
        filename,
        width = width,
        height = height,
        units = "in",
        res = 260,
        type = "cairo",
        ...
      )
    }
  )
}

calibrated <- read_csv(calibrated_path, show_col_types = FALSE) %>%
  arrange(trade_date) %>%
  mutate(
    trade_date = as.Date(trade_date),
    residual = simulated_price - actual_price,
    abs_error = abs(residual),
    naive_forecast = lag(actual_price),
    naive_error = naive_forecast - actual_price,
    squared_loss_diff = residual^2 - naive_error^2,
    error_side = if_else(residual >= 0, "模型偏高", "模型偏低")
  )

short_summary <- calibrated %>%
  filter(!is.na(naive_forecast)) %>%
  summarise(
    model_rmse = sqrt(mean(residual^2, na.rm = TRUE)),
    naive_rmse = sqrt(mean(naive_error^2, na.rm = TRUE)),
    rmse_improvement_pct = (1 - model_rmse / naive_rmse) * 100,
    mean_abs_error = mean(abs_error, na.rm = TRUE)
  )

short_plot <- ggplot(calibrated, aes(x = trade_date, y = residual)) +
  geom_hline(yintercept = 0, linewidth = 0.45, colour = "#111827") +
  geom_col(aes(fill = error_side), width = 0.72, alpha = 0.78) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.9, colour = "#111827", span = 0.42) +
  scale_fill_manual(values = c("模型偏高" = "#dc2626", "模型偏低" = "#2563eb")) +
  scale_x_date(labels = date_label_cn, breaks = breaks_width("2 weeks")) +
  labs(
    title = "短期模型残差的时间结构",
    subtitle = paste0(
      "R/ggplot2 复核：模型 RMSE=", fmt(short_summary$model_rmse),
      "，朴素基准 RMSE=", fmt(short_summary$naive_rmse),
      "，改善率=", fmt(short_summary$rmse_improvement_pct), "%"
    ),
    x = NULL,
    y = "模拟价 - 真实价（美元/桶）",
    caption = "数据来源：附件冲突窗口与 Python 主模型输出；R 仅用于图表复核。"
  ) +
  paper_theme()
save_plot(short_plot, short_error_fig)

state_quantile <- read_csv(state_quantile_path, show_col_types = FALSE) %>%
  mutate(trade_date = as.Date(trade_date))
scenario_neutral <- read_csv(scenario_path, show_col_types = FALSE) %>%
  mutate(trade_date = as.Date(trade_date)) %>%
  filter(scenario == "neutral", is_observed_price == FALSE)

final_state <- state_quantile %>% filter(day_index == 180)
state_plot <- ggplot(state_quantile, aes(x = trade_date)) +
  annotate("rect", xmin = min(state_quantile$trade_date), xmax = max(state_quantile$trade_date), ymin = 110, ymax = 120, fill = "#f59e0b", alpha = 0.12) +
  geom_ribbon(aes(ymin = p05, ymax = p95), fill = "#bfdbfe", alpha = 0.72) +
  geom_ribbon(aes(ymin = p25, ymax = p75), fill = "#3b82f6", alpha = 0.34) +
  geom_line(aes(y = p50, colour = "状态转移中位数"), linewidth = 1.05) +
  geom_line(data = scenario_neutral, aes(y = forecast_price, colour = "原中性中心线"), linewidth = 0.85, linetype = "22") +
  geom_hline(yintercept = 120, colour = "#b91c1c", linewidth = 0.45, linetype = "dotted") +
  scale_colour_manual(values = c("状态转移中位数" = "#1d4ed8", "原中性中心线" = "#111827")) +
  scale_x_date(labels = date_label_cn, breaks = breaks_width("3 weeks")) +
  labs(
    title = "长期状态转移区间预测",
    subtitle = paste0(
      "第180天 P05-P95：", fmt(final_state$p05), "-",
      fmt(final_state$p95), " 美元/桶；中位数：", fmt(final_state$p50), " 美元/桶"
    ),
    x = NULL,
    y = "美元/桶",
    caption = "阴影表示条件路径区间；中心线不是未来每日真实价格。"
  ) +
  paper_theme()
save_plot(state_plot, state_fan_fig, width = 10, height = 5.8)

baseline_dist <- read_csv(history_baseline_path, show_col_types = FALSE) %>%
  mutate(
    基准模型 = gsub("_RMSE", "", 基准指标),
    基准模型 = factor(基准模型, levels = 基准模型)
  )

history_plot <- ggplot(baseline_dist, aes(y = 基准模型)) +
  geom_linerange(aes(xmin = 历史P05, xmax = 历史P95), linewidth = 7.5, colour = "#dbeafe", alpha = 0.88) +
  geom_linerange(aes(xmin = 历史P25, xmax = 历史P75), linewidth = 7.5, colour = "#60a5fa", alpha = 0.72) +
  geom_point(aes(x = 历史中位数), size = 3.4, colour = "#1d4ed8") +
  geom_point(aes(x = 冲突窗口数值), size = 3.8, colour = "#dc2626") +
  geom_text(aes(x = 冲突窗口数值, label = paste0("冲突窗口 ", fmt(冲突窗口数值))), hjust = -0.08, size = 3.4, colour = "#991b1b", family = chinese_font) +
  labs(
    title = "历史同长度窗口的基准误差分布",
    subtitle = "冲突窗口相对 2017-2026 历史样本显著更难预测",
    x = "RMSE（美元/桶）",
    y = NULL,
    caption = "浅蓝为历史 P05-P95，深蓝为 P25-P75，蓝点为历史中位数，红点为 2026 冲突窗口。"
  ) +
  coord_cartesian(xlim = c(0, max(baseline_dist$冲突窗口数值, na.rm = TRUE) * 1.18)) +
  paper_theme()
save_plot(history_plot, history_baseline_fig, width = 9.8, height = 4.8)

summary <- tibble::tibble(
  item = c(
    "short_model_rmse",
    "naive_rmse",
    "rmse_improvement_pct",
    "state_day180_p05",
    "state_day180_p50",
    "state_day180_p95",
    "history_naive_event_percentile"
  ),
  value = c(
    short_summary$model_rmse,
    short_summary$naive_rmse,
    short_summary$rmse_improvement_pct,
    final_state$p05,
    final_state$p50,
    final_state$p95,
    baseline_dist$冲突窗口历史分位数[baseline_dist$基准指标 == "朴素上一日基准_RMSE"]
  )
)
write_csv(summary, summary_csv)

report <- paste0(
  "# R学术图表增强报告\n\n",
  "## 定位\n\n",
  "本脚本使用 R + ggplot2 对论文关键图进行学术化重绘，不改变 Python 主模型、参数校准或预测结果。它的价值是提升论文图表统一性、可读性和统计表达质感。\n\n",
  "## 新增图表\n\n",
  "- `figures/R短期误差学术诊断.png`：用残差柱形和 LOESS 趋势展示短期模型误差结构。\n",
  "- `figures/R长期状态转移扇形图.png`：用双层置信带展示长期状态转移区间，弱化单条中性线的误读。\n",
  "- `figures/R历史基准误差分布图.png`：用分位带展示冲突窗口相对历史同长度窗口的预测难度。\n\n",
  "## 关键数值\n\n",
  "- 短期模型 RMSE 为 ", fmt(short_summary$model_rmse), "，朴素基准 RMSE 为 ", fmt(short_summary$naive_rmse),
  "，改善率为 ", fmt(short_summary$rmse_improvement_pct), "%。\n",
  "- 状态转移第 180 天 P05-P95 为 ", fmt(final_state$p05), "-", fmt(final_state$p95), " 美元/桶。\n",
  "- 朴素上一日基准在冲突窗口的历史分位数约为 ",
  fmt(baseline_dist$冲突窗口历史分位数[baseline_dist$基准指标 == "朴素上一日基准_RMSE"] * 100, 1), "%。\n\n",
  "## 论文使用建议\n\n",
  "R 图表适合放在模型检验、长期预测和历史稳健性部分。它们不宣称模型更准，但能把“为什么可信、为什么不应只看一条线”的证据讲得更清楚。\n"
)
writeLines(report, report_path, useBytes = TRUE)

cat("R academic ggplot figures complete\n")
cat(short_error_fig, "\n")
cat(state_fan_fig, "\n")
cat(history_baseline_fig, "\n")
cat(report_path, "\n")
