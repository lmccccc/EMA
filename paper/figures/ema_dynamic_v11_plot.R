library(ggplot2)
library(patchwork)
library(tikzDevice)

# The published trace uses --timing=total; the default scrub view requires measurements.
# An optional --output-dir=existing-directory stages figures without replacing paper assets.
file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(file_arg) != 1) {
  stop("Run this figure generator with Rscript.")
}
figure_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))
source(file.path(figure_dir, "paper_plot_style.R"))

args <- commandArgs(trailingOnly = TRUE)
timing_arg <- grep("^--timing=", args, value = TRUE)
output_arg <- grep("^--output-dir=", args, value = TRUE)
if (length(timing_arg) > 1 || length(output_arg) > 1 ||
    length(args) != length(timing_arg) + length(output_arg)) {
  stop("Usage: Rscript ema_dynamic_v11_plot.R [--timing=total|scrub] [--output-dir=existing-directory]")
}
timing_mode <- if (length(timing_arg)) sub("^--timing=", "", timing_arg) else "scrub"
if (!(timing_mode %in% c("total", "scrub"))) {
  stop("--timing must be total or scrub.")
}
show_scrub <- timing_mode == "scrub"
output_dir <- if (length(output_arg)) {
  normalizePath(sub("^--output-dir=", "", output_arg), mustWork = TRUE)
} else {
  figure_dir
}
if (!dir.exists(output_dir)) {
  stop("--output-dir must name an existing directory.")
}

data <- read.csv(
  file.path(figure_dir, "data", "ema_dynamic_v11.csv"),
  stringsAsFactors = FALSE,
  na.strings = c("", "NA"),
  check.names = FALSE
)
operations <- c("insert", "delete", "point_update")
target_recall <- 0.95
expected_keys <- as.vector(outer(operations, 0:5, paste, sep = ":"))
actual_keys <- paste(data$operation, data$stage, sep = ":")
stopifnot(
  nrow(data) == 18,
  "qps95" %in% names(data),
  !("qps90" %in% names(data)),
  !anyDuplicated(actual_keys),
  setequal(actual_keys, expected_keys),
  all(data$progress == data$stage * 1e6),
  all(data$recall >= target_recall),
  all(data$ef >= 10)
)

data$Operation <- factor(
  data$operation,
  levels = operations,
  labels = c("Insert", "Delete", "Update")
)
operation_colours <- c(
  "Insert" = paper_colours[["EMA"]],
  "Delete" = paper_colours[["NaviX"]],
  "Update" = paper_colours[["Milvus"]]
)
operation_shapes <- c("Insert" = 16, "Delete" = 17, "Update" = 15)
scrub_colour <- "#D8D8D8"

qps95 <- data[!is.na(data$qps95), ]
minimum_ef <- data[is.na(data$qps95), ]
stopifnot(
  nrow(qps95) + nrow(minimum_ef) == 18,
  all(minimum_ef$ef == 10),
  all(minimum_ef$recall > target_recall),
  all(is.finite(minimum_ef$observed_qps)),
  all(is.finite(qps95$qps95)),
  all(qps95$qps95 > 0)
)

# Keep the metric distinction in the data and paper text, not in the curve styling.
data$qps_point_type <- ifelse(is.na(data$qps95), "observed_minimum_ef", "qps95")
data$plotted_qps <- ifelse(
  is.na(data$qps95),
  data$observed_qps,
  data$qps95
)
qps_breaks <- pretty(range(data$plotted_qps) * c(0.97, 1.03), n = 3)
# Both panels share these stage coordinates; stage 0 has no maintenance observation.
stage_range <- c(-0.5, 5.5)

dynamic_theme <- paper_theme(base_size = 8) +
  theme(
    axis.title = element_text(face = "bold", size = 8),
    axis.text = element_text(size = 8, colour = "black"),
    legend.text = element_text(size = 8),
    legend.spacing.x = unit(0.07, "cm"),
    legend.margin = margin(0, 0, 1, 0),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(1, 2, 1, 1)
  )

qps_plot <- ggplot(
  data,
  aes(x = stage, y = plotted_qps, group = Operation, colour = Operation, shape = Operation)
) +
  geom_line(linewidth = 0.65, linetype = "solid") +
  geom_paper_point(size = 1.8, stroke = 0.4) +
  scale_colour_manual(values = operation_colours, drop = FALSE) +
  scale_shape_manual(values = operation_shapes, drop = FALSE) +
  scale_x_continuous(breaks = 0:5, limits = stage_range, expand = expansion(mult = 0)) +
  scale_y_continuous(
    breaks = qps_breaks,
    limits = range(qps_breaks),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(x = "Cumulative changes (million)", y = "QPS", colour = NULL, shape = NULL) +
  dynamic_theme +
  guides(
    colour = guide_legend(
      title = NULL, nrow = 1,
      keywidth = unit(0.45, "cm"),
      keyheight = unit(0.25, "cm"),
      override.aes = list(linewidth = 0.65, size = 1.8)
    )
  )

cost <- data[data$stage > 0, ]
cost$maintenance_minutes <- cost$maintenance_seconds / 60
with_deletion <- cost$operation %in% c("delete", "point_update")
stopifnot(
  nrow(cost) == 15,
  all(cost$maintenance_records == 1e6),
  all(is.finite(cost$maintenance_minutes)),
  all(cost$maintenance_minutes > 0),
  all(abs(
    cost$maintenance_seconds[with_deletion] -
      cost$delete_seconds[with_deletion] -
      ifelse(cost$operation[with_deletion] == "point_update",
             cost$add_seconds[with_deletion], 0)
  ) < 1e-7)
)
if (show_scrub) {
  if (!("delete_scrub_seconds" %in% names(cost))) {
    stop("Figure 7(b) needs measured same-batch scrub durations; historical totals cannot be split. Use --timing=total for unsplit original totals.")
  }
  stopifnot(
    all(is.finite(cost$delete_scrub_seconds[with_deletion])),
    all(cost$delete_scrub_seconds[with_deletion] >= 0),
    all(cost$delete_scrub_seconds[with_deletion] <= cost$delete_seconds[with_deletion])
  )
  cost$scrub_minutes <- ifelse(with_deletion, cost$delete_scrub_seconds / 60, 0)
  cost$non_scrub_minutes <- cost$maintenance_minutes - cost$scrub_minutes
}
cost$bar_x <- cost$stage + (as.integer(cost$Operation) - 2) * 0.8 / 3
cost$xmin <- cost$bar_x - 0.72 / 6
cost$xmax <- cost$bar_x + 0.72 / 6

segments <- data.frame(
  xmin = cost$xmin, xmax = cost$xmax, ymin = 0,
  ymax = if (show_scrub) cost$non_scrub_minutes else cost$maintenance_minutes,
  Fill = as.character(cost$Operation)
)
fill_colours <- operation_colours
if (show_scrub) {
  segments <- rbind(
    segments,
    data.frame(
      xmin = cost$xmin[with_deletion], xmax = cost$xmax[with_deletion],
      ymin = cost$non_scrub_minutes[with_deletion],
      ymax = cost$maintenance_minutes[with_deletion], Fill = "Scrub"
    )
  )
  fill_colours <- c(fill_colours, "Scrub" = scrub_colour)
  scrub_labels <- cost[with_deletion, ]
  scrub_labels$label <- paste0(
    formatC(scrub_labels$delete_scrub_seconds, digits = 2, format = "fg", flag = "#"),
    "s"
  )
}
segments$Fill <- factor(segments$Fill, levels = names(fill_colours))
cost_tick_top <- max(8, ceiling(max(cost$maintenance_minutes) / 2) * 2)

cost_plot <- ggplot() +
  geom_rect(
    data = segments,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = Fill),
    colour = NA
  ) +
  geom_rect(
    data = cost,
    aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = maintenance_minutes),
    fill = NA,
    colour = "black",
    linewidth = 0.18
  ) +
  scale_fill_manual(
    values = fill_colours,
    drop = FALSE
  ) +
  scale_x_continuous(breaks = 0:5, limits = stage_range, expand = expansion(mult = 0)) +
  scale_y_continuous(
    breaks = seq(0, cost_tick_top, by = 2),
    limits = c(0, cost_tick_top + if (show_scrub) 2.5 else 0.4),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(x = "Stage (1 million changes each)", y = "Time per stage (min)", fill = NULL) +
  dynamic_theme +
  guides(
    fill = guide_legend(
      title = NULL, nrow = 1,
      keywidth = unit(0.35, "cm"),
      keyheight = unit(0.25, "cm")
    )
  )
if (show_scrub) {
  cost_plot <- cost_plot +
    geom_text(
      data = scrub_labels,
      aes(x = bar_x, y = maintenance_minutes + 0.12, label = label),
      angle = 90, hjust = 0, vjust = 0.5, size = 1.7,
      colour = "black"
    )
}

qps_compact <- qps_plot +
  labs(x = "Stage") +
  annotate(
    "text", x = -Inf, y = Inf, label = "(b)",
    hjust = -0.2, vjust = 1.2, size = 8 / .pt,
    fontface = "bold", family = "serif"
  ) +
  theme(plot.margin = margin(4, 1, 0, 1))
cost_compact <- cost_plot +
  labs(x = NULL, y = "Time (min)") +
  annotate(
    "text", x = -Inf, y = Inf, label = "(a)",
    hjust = -0.2, vjust = 1.2, size = 8 / .pt,
    fontface = "bold", family = "serif"
  ) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.x = element_blank(),
    plot.margin = margin(0, 1, 4, 1)
  )
if (show_scrub) {
  cost_compact <- cost_compact +
    scale_fill_manual(values = fill_colours, breaks = "Scrub", drop = FALSE)
} else {
  cost_compact <- cost_compact + guides(fill = "none")
}
combined_plot <- (cost_compact / qps_compact) +
  plot_layout(guides = "collect", heights = c(1, 1)) +
  plot_annotation(theme = theme(plot.margin = margin(0, 0, 0, 0))) &
  theme(legend.position = "top", legend.box = "horizontal")

qps_build <- ggplot_build(qps_compact)
cost_build <- ggplot_build(cost_compact)
qps_layers <- qps_build$data
delete_points <- qps_layers[[2]][
  qps_layers[[2]]$colour == operation_colours[["Delete"]], ]
cost_layers <- cost_build$data
bar_layer <- cost_layers[[1]]
outline_layer <- cost_layers[[2]]
stopifnot(
  length(qps_layers) == 3,
  nrow(qps_layers[[2]]) == 18,
  all(qps_layers[[1]]$linetype == "solid"),
  nrow(delete_points) == 6,
  all(delete_points$shape == operation_shapes[["Delete"]]),
  all(sort(qps_layers[[2]]$y) == sort(data$plotted_qps)),
  identical(qps_layers[[3]]$label, "(b)"),
  length(cost_layers) == if (show_scrub) 4 else 3,
  identical(tail(cost_layers, 1)[[1]]$label, "(a)"),
  nrow(bar_layer) == if (show_scrub) 25 else 15,
  all(bar_layer$ymax >= bar_layer$ymin),
  all(abs(bar_layer$ymax - bar_layer$ymin - (segments$ymax - segments$ymin)) < 1e-10),
  nrow(outline_layer) == 15,
  all(outline_layer$ymin == 0),
  all(sort(outline_layer$ymax) == sort(cost$maintenance_minutes)),
  all(cost$stage %in% 1:5),
  identical(
    qps_build$layout$panel_params[[1]]$x.range,
    cost_build$layout$panel_params[[1]]$x.range
  )
)
if (show_scrub) {
  stopifnot(
    nrow(cost_layers[[3]]) == 10,
    all(cost_layers[[3]]$label == scrub_labels$label)
  )
}
check_shared_layout <- function(plot) {
  # Measuring grobs without an explicit device would create Rplots.pdf.
  grDevices::pdf(file = NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  layout <- patchworkGrob(plot)$layout
  panels <- layout[layout$name %in% c("panel-1", "panel-2"), ]
  stopifnot(
    nrow(panels) == 2,
    length(unique(panels$l)) == 1,
    length(unique(panels$r)) == 1,
    panels$t[panels$name == "panel-1"] < panels$t[panels$name == "panel-2"]
  )
}
check_shared_layout(combined_plot)

export_tikz <- function(plot, filename, width = 3.15, height = 1.65) {
  tikz(
    file = file.path(output_dir, filename),
    width = width,
    height = height,
    standAlone = FALSE,
    documentDeclaration = "\\documentclass[conference]{IEEEtran}",
    packages = c(getOption("tikzLatexPackages"), "\\usepackage[T1]{fontenc}")
  )
  on.exit(dev.off(), add = TRUE)
  print(plot)
}

export_tikz(qps_plot, "ema_dynamic_qps_tikz.tex")
export_tikz(cost_plot, "ema_dynamic_maintenance_tikz.tex")
# The published view sits beside the wider Table V; scrub labels need a larger canvas.
export_tikz(
  combined_plot, "ema_dynamic_compact_tikz.tex",
  width = if (show_scrub) 3.45 else 2.3,
  height = if (show_scrub) 2.6 else 1.6
)

plot_values <- data[c(
  "operation", "stage", "progress", "live", "qps95", "observed_qps",
  "recall", "ef", "maintenance_seconds",
  if (show_scrub) "delete_scrub_seconds",
  "qps_point_type", "plotted_qps"
)]
plot_values$plotted_maintenance_minutes <- ifelse(
  data$stage > 0, data$maintenance_seconds / 60, NA_real_
)
if (show_scrub) {
  plot_values$plotted_scrub_minutes <- ifelse(
    data$stage == 0, NA_real_,
    ifelse(data$operation == "insert", 0, data$delete_scrub_seconds / 60)
  )
  plot_values$plotted_non_scrub_minutes <-
    plot_values$plotted_maintenance_minutes - plot_values$plotted_scrub_minutes
}
output_data_dir <- file.path(output_dir, "data")
if (!dir.exists(output_data_dir) && !dir.create(output_data_dir)) {
  stop("Could not create the plot-values output directory.")
}
write.csv(
  plot_values,
  file.path(output_data_dir, "ema_dynamic_v11_plot_values.csv"),
  row.names = FALSE,
  na = ""
)
message("Generated the shared-axis dynamic figure (timing=", timing_mode,
        ") and individual plot views in ", output_dir, ".")
