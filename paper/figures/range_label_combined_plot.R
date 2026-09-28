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

high_env <- new.env(parent = globalenv())
low_env <- new.env(parent = globalenv())
composite_env <- new.env(parent = globalenv())

sys.source("range_label_high_sel_recall_plot.R", envir = high_env)
sys.source("range_label_sel_recall_plot.R", envir = low_env)
sys.source("range_label_composite_sel_recall_plot.R", envir = composite_env)

source("paper_plot_style.R")

legend_breaks <- c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "90% Recall")
scale_limits <- c(
  "EMA", "EMA_90",
  "NaviX", "NaviX_90",
  "ACORN", "ACORN_90",
  "VBase", "VBase_90",
  "Milvus", "90% Recall"
)
legend_labels <- c(
  "EMA" = "EMA",
  "NaviX" = "NaviX",
  "ACORN" = "ACORN",
  "VBase" = "VBase",
  "Milvus" = "Milvus(MT)",
  "90% Recall" = "90\\% Recall"
)
line_values <- c(
  "EMA" = "solid",
  "EMA_90" = "dashed",
  "NaviX" = "solid",
  "NaviX_90" = "dashed",
  "ACORN" = "solid",
  "ACORN_90" = "dashed",
  "VBase" = "solid",
  "VBase_90" = "dashed",
  "Milvus" = "solid",
  "90% Recall" = "dashed"
)

group_theme <- theme(
  axis.title = element_text(face = "bold", size = 8),
  axis.text = element_text(size = 6.1, colour = "black"),
  strip.text = element_text(
    face = "bold",
    size = 6.5,
    margin = margin(b = 0.3)
  ),
  plot.caption = element_text(
    face = "bold",
    size = 7.6,
    hjust = 0.5,
    margin = margin(0)
  ),
  plot.caption.position = "plot",
  legend.text = element_text(size = 7.3),
  legend.spacing.x = unit(0.10, "cm"),
  panel.spacing = unit(0.06, "cm"),
  panel.grid.major.x = element_blank(),
  panel.grid.minor = element_blank(),
  plot.margin = margin(0.5, 1.5, 0.5, 1.5)
)

build_group <- function(plot_data, title, x_scale, y_scales) {
  ggplot(
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
    ggh4x::facet_wrap2(
      ~Dataset,
      ncol = 2,
      scales = "free_y",
      axes = "all"
    ) +
    geom_line(linewidth = 0.6) +
    geom_paper_point(size = 1.8, stroke = 0.4) +
    scale_shape_manual(
      values = paper_shapes,
      breaks = legend_breaks,
      labels = legend_labels[legend_breaks],
      limits = scale_limits,
      drop = FALSE
    ) +
    scale_colour_manual(
      values = paper_colours,
      breaks = legend_breaks,
      labels = legend_labels[legend_breaks],
      limits = scale_limits,
      drop = FALSE
    ) +
    scale_linetype_manual(
      values = line_values,
      breaks = legend_breaks,
      labels = legend_labels[legend_breaks],
      limits = scale_limits,
      drop = FALSE
    ) +
    x_scale +
    ggh4x::facetted_pos_scales(y = y_scales) +
    labs(caption = title, x = "Selectivity (\\%)", y = "QPS") +
    paper_theme(base_size = 7.2) +
    group_theme +
    guides(
      shape = paper_guide(nrow = 1, key_width = 0.48),
      colour = paper_guide(nrow = 1, key_width = 0.48),
      linetype = paper_guide(nrow = 1, key_width = 0.48)
    )
}

high_plot <- build_group(
  high_env$data,
  "(a) Label+range (10\\%--100\\%)",
  scale_x_continuous(
    labels = function(x) format(x * 100, trim = TRUE),
    breaks = c(0.1, 0.4, 0.6, 0.8, 1)
  ),
  list(
    Dataset == "Redcaps4M" ~ scale_y_continuous(
      limits = c(0, 3600),
      breaks = seq(0, 3600, by = 1200),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "SIFT10M" ~ scale_y_continuous(
      limits = c(0, 2100),
      breaks = seq(0, 1800, by = 600),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "YoutubeRGB1M" ~ scale_y_continuous(
      limits = c(0, 650),
      breaks = seq(0, 600, by = 200),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "Wiki15.4M" ~ scale_y_continuous(
      limits = c(0, 1600),
      breaks = seq(0, 1600, by = 400),
      expand = expansion(mult = c(0.05, 0.02))
    )
  )
)

low_plot <- build_group(
  low_env$data,
  "(b) Label+range (1\\%--10\\%)",
  scale_x_log10(
    labels = function(x) format(x * 100, trim = TRUE),
    breaks = c(0.01, 0.02, 0.05, 0.1)
  ),
  list(
    Dataset == "Redcaps4M" ~ scale_y_continuous(
      limits = c(0, 1800),
      breaks = seq(0, 1800, by = 600),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "SIFT10M" ~ scale_y_continuous(
      limits = c(0, 1400),
      breaks = seq(0, 1200, by = 400),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "YoutubeRGB1M" ~ scale_y_continuous(
      limits = c(0, 400),
      breaks = seq(0, 400, by = 100),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "Wiki15.4M" ~ scale_y_continuous(
      limits = c(0, 900),
      breaks = seq(0, 800, by = 200),
      expand = expansion(mult = c(0.05, 0.02))
    )
  )
)

composite_plot <- build_group(
  composite_env$data,
  "(c) Composed multi-predicate (1\\%--10\\%)",
  scale_x_log10(
    labels = function(x) format(x * 100, trim = TRUE),
    breaks = c(0.01, 0.02, 0.05, 0.1)
  ),
  list(
    Dataset == "Redcaps4M" ~ scale_y_continuous(
      limits = c(0, 1600),
      breaks = seq(0, 1600, by = 400),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "SIFT10M" ~ scale_y_continuous(
      limits = c(0, 1600),
      breaks = seq(0, 1600, by = 400),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "YoutubeRGB1M" ~ scale_y_continuous(
      limits = c(0, 400),
      breaks = seq(0, 400, by = 100),
      expand = expansion(mult = c(0.05, 0.02))
    ),
    Dataset == "Wiki15.4M" ~ scale_y_continuous(
      limits = c(0, 1100),
      breaks = seq(0, 1000, by = 250),
      expand = expansion(mult = c(0.05, 0.02))
    )
  )
)

extract_top_legend <- function(plot) {
  plot_grob <- ggplotGrob(plot + theme(legend.position = "top"))
  legend_index <- which(plot_grob$layout$name == "guide-box-top")
  plot_grob$grobs[[legend_index]]
}

legend_grob <- extract_top_legend(high_plot)
body_plot <- (
  (high_plot + theme(legend.position = "none")) |
    (low_plot + theme(legend.position = "none")) |
    (composite_plot + theme(legend.position = "none"))
) +
  plot_layout(
    axis_titles = "collect",
    widths = c(1, 1, 1)
  )

combined_plot <- wrap_elements(full = legend_grob) / body_plot +
  plot_layout(heights = c(0.13, 1))

tikz(
  file = "range_label_combined_qps_tikz.tex",
  width = 7.0,
  height = 2.75,
  standAlone = FALSE
)
print(combined_plot)
while (dev.cur() > 1) dev.off()
