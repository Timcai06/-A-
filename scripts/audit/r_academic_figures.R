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
ink <- "#1A1A1A"
lime <- "#55ff37"
emerald <- "#00e978"
teal <- "#00cfae"
cyan <- "#00b3d9"
sky <- "#0095f2"
blue <- "#0074f2"
muted <- "#4b5563"
grid <- "#E5E7EB"
border <- "#CBD5E1"
band_light <- "#DDF8FF"
band <- "#99E7F4"

calibrated_path <- file.path(root, "output", "calibration", "动态模型校准后路径.csv")
history_baseline_path <- file.path(root, "output", "history", "历史基准误差分布.csv")
state_quantile_path <- file.path(root, "output", "scenarios", "长期状态转移路径分位数.csv")
scenario_path <- file.path(root, "output", "scenarios", "三情景预测结果.csv")
sensitivity_path <- file.path(root, "output", "sensitivity", "参数重要性排序.csv")
candidates_path <- file.path(root, "output", "calibration", "动态模型候选参数前10.csv")
risk_sample_path <- file.path(root, "output", "risk", "市场定价风险变量_滞后检验样本.csv")

short_error_fig <- file.path(fig_dir, "R短期误差学术诊断.png")
state_fan_fig <- file.path(fig_dir, "R长期状态转移扇形图.png")
history_baseline_fig <- file.path(fig_dir, "R历史基准误差分布图.png")
r_sensitivity_fig <- file.path(fig_dir, "R参数敏感性龙卷风图.png")
r_candidate_fig <- file.path(fig_dir, "R候选模型误差对比.png")
r_risk_fig <- file.path(fig_dir, "R市场风险变量滞后检验.png")
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
      plot.title = element_text(family = chinese_font, face = "bold", colour = ink, size = base_size + 2),
      plot.subtitle = element_text(family = chinese_font, colour = muted, size = base_size - 1),
      axis.title = element_text(family = chinese_font, colour = muted),
      axis.text = element_text(family = chinese_font, colour = muted),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = grid, linewidth = 0.35),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(family = chinese_font, colour = muted, size = base_size - 3),
      legend.key.size = unit(0.42, "lines"),
      legend.spacing.x = unit(0.25, "lines"),
      legend.box.margin = margin(0, 0, 2, 0),
      legend.background = element_blank(),
      plot.caption = element_text(family = chinese_font, colour = muted, size = base_size - 3)
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
  geom_hline(yintercept = 0, linewidth = 0.45, colour = ink) +
  geom_col(aes(fill = error_side), width = 0.72, alpha = 0.78) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.9, colour = ink, span = 0.42) +
  scale_fill_manual(values = c("模型偏高" = sky, "模型偏低" = teal)) +
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
  annotate("rect", xmin = min(state_quantile$trade_date), xmax = max(state_quantile$trade_date), ymin = 110, ymax = 120, fill = teal, alpha = 0.08) +
  geom_ribbon(aes(ymin = p05, ymax = p95), fill = band_light, alpha = 0.72) +
  geom_ribbon(aes(ymin = p25, ymax = p75), fill = band, alpha = 0.34) +
  geom_line(aes(y = p50, colour = "状态转移中位数"), linewidth = 1.05) +
  geom_line(data = scenario_neutral, aes(y = forecast_price, colour = "原中性中心线"), linewidth = 0.85, linetype = "22") +
  geom_hline(yintercept = 120, colour = sky, linewidth = 0.45, linetype = "dotted") +
  scale_colour_manual(values = c("状态转移中位数" = blue, "原中性中心线" = ink)) +
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
  geom_linerange(aes(xmin = 历史P05, xmax = 历史P95), linewidth = 7.5, colour = band_light, alpha = 0.88) +
  geom_linerange(aes(xmin = 历史P25, xmax = 历史P75), linewidth = 7.5, colour = band, alpha = 0.72) +
  geom_point(aes(x = 历史中位数), size = 3.4, colour = blue) +
  geom_point(aes(x = 冲突窗口数值), size = 3.8, colour = sky) +
  geom_text(aes(x = 冲突窗口数值, label = paste0("冲突窗口 ", fmt(冲突窗口数值))), hjust = -0.08, size = 3.4, colour = sky, family = chinese_font) +
  labs(
    title = "历史同长度窗口的基准误差分布",
    subtitle = "冲突窗口相对 2017-2026 历史样本显著更难预测",
    x = "RMSE（美元/桶）",
    y = NULL,
    caption = "浅蓝为历史 P05-P95，深蓝为 P25-P75，蓝点为历史中位数，亮蓝点为 2026 冲突窗口。"
  ) +
  coord_cartesian(xlim = c(0, max(baseline_dist$冲突窗口数值, na.rm = TRUE) * 1.18)) +
  paper_theme()
save_plot(history_plot, history_baseline_fig, width = 9.8, height = 4.8)

if (file.exists(sensitivity_path)) {
  sensitivity <- read_csv(sensitivity_path, show_col_types = FALSE) %>%
    arrange(desc(综合敏感度得分)) %>%
    slice_head(n = 8) %>%
    mutate(
      参数 = factor(参数, levels = rev(参数)),
      风险层 = if_else(可控性 == "低", "外生低可控", "部分可控")
    )

  sensitivity_plot <- ggplot(sensitivity, aes(x = 综合敏感度得分, y = 参数, fill = 风险层)) +
    geom_col(width = 0.66, alpha = 0.92) +
    geom_text(aes(label = fmt(综合敏感度得分, 1)), hjust = -0.12, size = 3.4, family = chinese_font, colour = ink) +
    scale_fill_manual(values = c("外生低可控" = blue, "部分可控" = teal)) +
    coord_cartesian(xlim = c(0, max(sensitivity$综合敏感度得分, na.rm = TRUE) * 1.18)) +
    labs(
      title = "长期参数敏感性的主导因素",
      subtitle = "封锁风险衰减、地缘风险权重和制度风险项是长期价格区间的主要扰动源",
      x = "综合敏感度得分",
      y = NULL,
      caption = "深蓝表示外生低可控变量，青绿色表示可通过模型机制或政策假设部分约束的变量。"
    ) +
    paper_theme(base_size = 11)
  save_plot(sensitivity_plot, r_sensitivity_fig, width = 9.8, height = 5.4)
}

if (file.exists(candidates_path)) {
  candidates <- read_csv(candidates_path, show_col_types = FALSE) %>%
    slice_head(n = 6) %>%
    mutate(
      候选标签 = paste0("候选", candidate_id),
      候选标签 = factor(候选标签, levels = 候选标签)
    )
  candidate_metrics <- bind_rows(
    candidates %>% transmute(候选标签, 指标 = "RMSE", 误差 = RMSE),
    candidates %>% transmute(候选标签, 指标 = "高价平台RMSE", 误差 = 高价平台RMSE),
    candidates %>% transmute(候选标签, 指标 = "低价回落RMSE", 误差 = 低价回落RMSE)
  )

  candidate_plot <- ggplot(candidate_metrics, aes(x = 候选标签, y = 误差, fill = 指标)) +
    geom_col(position = position_dodge(width = 0.76), width = 0.68, alpha = 0.92) +
    scale_fill_manual(values = c("RMSE" = blue, "高价平台RMSE" = teal, "低价回落RMSE" = sky)) +
    labs(
      title = "短期候选模型误差结构对比",
      subtitle = "综合最优候选不是单看一个指标取胜，而是在整体、平台和回落段之间取得平衡",
      x = NULL,
      y = "误差（美元/桶）",
      caption = "数据来源：动态模型候选参数前10；这里只展示前6个代表候选。"
    ) +
    paper_theme(base_size = 11) +
    theme(axis.text.x = element_text(angle = 0, vjust = 0.6))
  save_plot(candidate_plot, r_candidate_fig, width = 10.2, height = 5.4)
}

if (file.exists(risk_sample_path)) {
  risk_sample <- read_csv(risk_sample_path, show_col_types = FALSE) %>%
    mutate(trade_date = as.Date(trade_date)) %>%
    filter(trade_date >= as.Date("2025-01-01")) %>%
    filter(!is.na(ovx_lag1), !is.na(volatility_7d), !is.na(abs_return))

  scale01 <- function(x) {
    rng <- range(x, na.rm = TRUE)
    if (isTRUE(all.equal(rng[[1]], rng[[2]]))) return(rep(0.5, length(x)))
    (x - rng[[1]]) / (rng[[2]] - rng[[1]])
  }

  risk_long <- bind_rows(
    risk_sample %>% transmute(trade_date, 指标 = "滞后OVX隐含波动率", 标准化数值 = scale01(ovx_lag1)),
    risk_sample %>% transmute(trade_date, 指标 = "7日实现波动率", 标准化数值 = scale01(volatility_7d)),
    risk_sample %>% transmute(trade_date, 指标 = "当日绝对收益", 标准化数值 = scale01(abs_return))
  )

  risk_plot <- ggplot(risk_long, aes(x = trade_date, y = 标准化数值, colour = 指标)) +
    annotate("rect", xmin = as.Date("2026-03-02"), xmax = as.Date("2026-05-05"), ymin = -Inf, ymax = Inf, fill = sky, alpha = 0.07) +
    geom_line(linewidth = 0.9, alpha = 0.92) +
    scale_colour_manual(values = c("滞后OVX隐含波动率" = blue, "7日实现波动率" = sky, "当日绝对收益" = muted)) +
    scale_x_date(labels = date_label_cn, breaks = breaks_width("2 months")) +
    labs(
      title = "市场定价风险变量的滞后可用性检验",
      subtitle = "滞后 OVX 与实现波动在冲突窗口同步抬升，可用于长期不确定性强度而非直接替代价格",
      x = NULL,
      y = "0-1 标准化数值",
      caption = "淡蓝区为冲突窗口；全部风险变量只使用滞后或可观测口径，避免当日价格反向解释当日风险。"
    ) +
    paper_theme(base_size = 11)
  save_plot(risk_plot, r_risk_fig, width = 10.5, height = 5.6)
}

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
  "- `figures/R历史基准误差分布图.png`：用分位带展示冲突窗口相对历史同长度窗口的预测难度。\n",
  "- `figures/R参数敏感性龙卷风图.png`：用横向排序图突出长期模型的关键敏感变量。\n",
  "- `figures/R候选模型误差对比.png`：用分组柱形图展示短期候选模型的误差权衡。\n",
  "- `figures/R市场风险变量滞后检验.png`：用标准化时间序列展示 OVX 等市场风险变量的滞后证据。\n\n",
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
if (file.exists(r_sensitivity_fig)) cat(r_sensitivity_fig, "\n")
if (file.exists(r_candidate_fig)) cat(r_candidate_fig, "\n")
if (file.exists(r_risk_fig)) cat(r_risk_fig, "\n")
cat(report_path, "\n")
