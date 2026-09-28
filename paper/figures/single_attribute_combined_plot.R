library(ggplot2)
library(ggh4x)
library(patchwork)
library(tikzDevice)

options(
  tikzLatexPackages = c(
    getOption("tikzLatexPackages"),
    "\\usepackage{graphicx}"
  )
)

Sys.setenv(EMA_SKIP_PLOT = "1")
on.exit(Sys.unsetenv("EMA_SKIP_PLOT"), add = TRUE)

range_env <- new.env(parent = globalenv())
label_env <- new.env(parent = globalenv())
ocq_env <- new.env(parent = globalenv())

sys.source("range_sel_recall_plot.R", envir = range_env)
sys.source("label_sel_recall_plot.R", envir = label_env)
sys.source("uncorr_sel.R", envir = ocq_env)

source("paper_plot_style.R")

method_breaks <- c(
  "EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "DiskANN"
)
method_labels <- c(
  "EMA" = "EMA",
  "NaviX" = "NaviX",
  "ACORN" = "ACORN",
  "VBase" = "VBase",
  "Milvus" = "Milvus(MT)",
  "iRangeGraph" = "iRangeGraph",
  "DiskANN" = "Filtered DiskANN"
)
scale_limits <- c(
  "EMA", "EMA_90",
  "NaviX", "NaviX_90",
  "ACORN", "ACORN_80", "ACORN_90",
  "VBase", "VBase_80", "VBase_90",
  "Milvus", "iRangeGraph",
  "DiskANN", "DiskANN_80",
  "90% Recall", "80% Recall"
)
line_values <- c(
  "EMA" = "solid",
  "EMA_90" = "dashed",
  "NaviX" = "solid",
  "NaviX_90" = "dashed",
  "ACORN" = "solid",
  "ACORN_80" = "dotted",
  "ACORN_90" = "dashed",
  "VBase" = "solid",
  "VBase_80" = "dotted",
  "VBase_90" = "dashed",
  "Milvus" = "solid",
  "iRangeGraph" = "solid",
  "DiskANN" = "solid",
  "DiskANN_80" = "dotted",
  "90% Recall" = "dashed",
  "80% Recall" = "dotted"
)

query_theme <- theme(
  axis.title = element_text(face = "bold", size = 8),
  axis.text = element_text(size = 6.2, colour = "black"),
  strip.text = element_text(
    face = "bold",
    size = 6.7,
    margin = margin(b = 0.3)
  ),
  plot.caption = element_text(
    face = "bold",
    size = 7.7,
    hjust = 0.5,
    margin = margin(0)
  ),
  plot.caption.position = "plot",
  panel.spacing = unit(0.07, "cm"),
  panel.grid.major.x = element_blank(),
  panel.grid.minor = element_blank(),
  plot.margin = margin(0.5, 1.5, 0.5, 1.5),
  legend.position = "none"
)

add_method_scales <- function(plot) {
  plot +
    scale_shape_manual(
      values = paper_shapes,
      breaks = method_breaks,
      labels = method_labels[method_breaks],
      limits = scale_limits,
      drop = FALSE
    ) +
    scale_colour_manual(
      values = paper_colours,
      breaks = method_breaks,
      labels = method_labels[method_breaks],
      limits = scale_limits,
      drop = FALSE
    ) +
    scale_linetype_manual(
      values = line_values,
      breaks = method_breaks,
      labels = method_labels[method_breaks],
      limits = scale_limits,
      drop = FALSE
    )
}

build_query_group <- function(
    plot_data,
    title,
    y_scales,
    x_breaks = c(0.01, 0.02, 0.05, 0.1),
    x_limits = NULL) {
  plot_data$Dataset <- factor(
    plot_data$Dataset,
    levels = c("Redcaps4M", "YoutubeRGB1M", "Wiki15.4M")
  )

  plot <- ggplot(
    plot_data,
    aes(
      x = x,
      y = y,
      group = Algorithm,
      color = Algorithm,
      shape = Algorithm,
      linetype = Algorithm
    )
  ) +
    ggh4x::facet_grid2(
      . ~ Dataset,
      scales = "free_y",
      axes = "all",
      independent = "y"
    ) +
    geom_line(linewidth = 0.6) +
    geom_paper_point(size = 1.8, stroke = 0.4) +
    scale_x_log10(
      labels = function(x) format(x * 100, trim = TRUE),
      breaks = x_breaks,
      limits = x_limits,
      expand = expansion(mult = c(0.03, 0.03))
    ) +
    ggh4x::facetted_pos_scales(y = y_scales) +
    labs(caption = title, x = "Selectivity (\\%)", y = "QPS") +
    paper_theme(base_size = 7.2) +
    query_theme

  add_method_scales(plot)
}

range_plot <- build_query_group(
  range_env$data,
  "(a) Range query",
  list(
    Dataset == "Redcaps4M" ~ scale_y_continuous(
      limits = c(0, 3600),
      breaks = seq(0, 3600, by = 1200),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "YoutubeRGB1M" ~ scale_y_continuous(
      limits = c(0, 1600),
      breaks = seq(0, 1600, by = 400),
      expand = expansion(mult = c(0.05, 0.02))
    )
  )
)

label_plot <- build_query_group(
  label_env$data,
  "(b) Label query",
  list(
    Dataset == "Redcaps4M" ~ scale_y_continuous(
      limits = c(0, 800),
      breaks = seq(0, 800, by = 200),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "YoutubeRGB1M" ~ scale_y_continuous(
      limits = c(0, 400),
      breaks = seq(0, 400, by = 100),
      expand = expansion(mult = c(0.05, 0.02))
    )
  )
)

ocq_plot <- build_query_group(
  ocq_env$data,
  "(c) OCQ",
  list(
    Dataset == "Wiki15.4M" ~ scale_y_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, by = 20),
      expand = expansion(mult = c(0.05, 0.02))
    )
  ),
  x_breaks = c(0.01, 0.05, 0.1, 0.2),
  x_limits = c(0.01, 0.23)
)

legend_data <- data.frame(
  x = seq_along(method_breaks),
  y = 1,
  Algorithm = factor(method_breaks, levels = scale_limits)
)
legend_plot <- add_method_scales(
  ggplot(
    legend_data,
    aes(
      x = x,
      y = y,
      group = Algorithm,
      color = Algorithm,
      shape = Algorithm,
      linetype = Algorithm
    )
  ) +
    geom_line(linewidth = 0.6) +
    geom_paper_point(size = 1.8, stroke = 0.4) +
    theme_void() +
    theme(
      legend.position = "top",
      legend.text = element_text(size = 7),
      legend.spacing.x = unit(0.08, "cm"),
      legend.margin = margin(0)
    ) +
    guides(
      shape = paper_guide(nrow = 1, key_width = 0.42),
      colour = paper_guide(nrow = 1, key_width = 0.42),
      linetype = paper_guide(nrow = 1, key_width = 0.42)
    )
)

legend_table <- ggplotGrob(legend_plot)
legend_grob <- legend_table$grobs[[
  which(legend_table$layout$name == "guide-box-top")
]]

body_plot <- (range_plot | label_plot | ocq_plot) +
  plot_layout(
    widths = c(2, 2, 1),
    axis_titles = "collect"
  )

combined_plot <- wrap_elements(full = legend_grob) / body_plot +
  plot_layout(heights = c(0.15, 1))

tikz(
  file = "single_attribute_combined_qps_tikz.tex",
  width = 7,
  height = 1.65,
  standAlone = FALSE
)
print(combined_plot)
while (dev.cur() > 1) dev.off()
