library(readxl)
library(scales)
library(tikzDevice)
library(patchwork)
library(reshape2)
library(plyr)
library(latex2exp)
library(tidyr)
# library(cowplot)     # 用于图形组合和图例提取
library(ggplot2)
library(ggh4x)
options(
  tikzLatexPackages = c(
    getOption('tikzLatexPackages'),
    "\\usepackage{graphicx}"
  )
)


remove_y <- function(){
  theme(
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
  )+ 
    theme(axis.title.y = element_blank())
}

# ----------------- youtube ------------------------

# bfann results
# Target recall: 0.95
bfann_youtube_rgb_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(595, 574, 550, 510, 431.34, 349.89)
)


# navix results
# Target recall: 0.95
navix_youtube_rgb_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(69.29, 63.12, 54.84, 42.77, 49.18, 28.73)
)



# acorn results
# Target recall: 0.95
acorn_youtube_rgb_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40),
  y = c(49.95, 56.16, 78.34, 84.28)
)

# milvus results
# Target recall: 0.95
milvus_youtube_rgb_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(77.95, 66.06, 61.44, 52.67, 58.72, 57.59)
)


legend_90_df <- data.frame(
  x = NA,
  y = NA,
  Algorithm = "90% Recall"
)
legend_90_df$Algorithm <- "90% Recall"

bfann_youtube_rgb_95_df$Algorithm <- "EMA"
navix_youtube_rgb_95_df$Algorithm <- "NaviX"
acorn_youtube_rgb_95_df$Algorithm <- "ACORN"
# msvbase_youtube_rgb_95_df$Algorithm <- "VBase"
milvus_youtube_rgb_95_df$Algorithm <- "Milvus"

youtube_data <- rbind(bfann_youtube_rgb_95_df, 
                      navix_youtube_rgb_95_df,  
                      acorn_youtube_rgb_95_df,
                      # msvbase_youtube_rgb_95_df,
                      milvus_youtube_rgb_95_df,
                      legend_90_df
)

# ----------------------sift -------------------
# bfann results
# Target recall: 0.95
bfann_sift10m_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(1986.99, 1857, 1754, 1609.19, 1472, 1348.73)
)


# navix results
# Target recall: 0.95
navix_sift10m_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(634.95, 542.82, 463, 326.55, 227.46, 148.30)
)


# acorn results
# Target recall: 0.95
acorn_sift10m_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(98.16, 111.11, 124.08, 86.24, 104.51, 36.45)
)

# msvbase results
# Target recall: 0.9
msvbase_sift10m_90_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40),
  y = c(834.24, 862.16, 846.09, 857.99)
)

# milvus results
# Target recall: 0.95
milvus_sift10m_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(18.61, 15.03, 13.69, 12.66, 13.54, 13.99)
)



bfann_sift10m_95_df$Algorithm <- "EMA"
navix_sift10m_95_df$Algorithm <- "NaviX"
acorn_sift10m_95_df$Algorithm <- "ACORN"
msvbase_sift10m_90_df$Algorithm <- "VBase_90"
milvus_sift10m_95_df$Algorithm <- "Milvus"

sift10m_data <- rbind(bfann_sift10m_95_df, 
                      navix_sift10m_95_df, 
                      acorn_sift10m_95_df,
                      msvbase_sift10m_90_df,
                      milvus_sift10m_95_df
)

# ------------------ redcaps ------------------------

# bfann results
# Target recall: 0.95
bfann_Redcaps_4M_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(3466, 3100, 2700, 2300, 1829.62, 1577)
)


# navix results
# Target recall: 0.95
navix_Redcaps_4M_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20,0.10),
  y = c(1495, 1292, 1100, 885, 645, 425.49)
)


# acorn results
# Target recall: 0.95
acorn_Redcaps_4M_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(589, 457.76, 261.80, 272.78, 117.79, 51.15)
)

# msvbase results
# Target recall: 0.95
msvbase_Redcaps_4M_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20),
  y = c(313.18, 267.03, 247.76, 302.01, 391.77)
)

# milvus results
# Target recall: 0.95
milvus_Redcaps_4M_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.1),
  y = c(34.54, 26.18, 22.80, 19.72, 22.73,30)
)


bfann_Redcaps_4M_95_df$Algorithm <- "EMA"
navix_Redcaps_4M_95_df$Algorithm <- "NaviX"
acorn_Redcaps_4M_95_df$Algorithm <- "ACORN"
msvbase_Redcaps_4M_95_df$Algorithm <- "VBase"
milvus_Redcaps_4M_95_df$Algorithm <- "Milvus"

redcaps_data <- rbind(bfann_Redcaps_4M_95_df, 
                      navix_Redcaps_4M_95_df,  
                      acorn_Redcaps_4M_95_df,
                      milvus_Redcaps_4M_95_df,
                      msvbase_Redcaps_4M_95_df
)


# -------------- wiki ----------------
# bfann results
# Target recall: 0.95
bfann_wiki_15_4M_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(1552, 1350, 1097.54, 860, 575.53, 507.95)
)

# navix results
# Target recall: 0.95
navix_wiki_15_4M_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(502, 464, 386.42, 278.55, 132.04, 39.05)
)

# acorn results
# Target recall: 0.95
acorn_wiki_15_4M_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20),
  y = c(77.31, 69.86, 138.47, 108.66, 93.49)
)

# msvbase results
# Target recall: 0.95
msvbase_wiki_15_4M_95_df <- data.frame(
  x = c(1.00, 0.80, 0.60, 0.40, 0.20, 0.10),
  y = c(15.70, 11.01, 7.53, 5.21, 7.38, 2.29)
)

bfann_wiki_15_4M_95_df$Algorithm <- "EMA"
navix_wiki_15_4M_95_df$Algorithm <- "NaviX"
acorn_wiki_15_4M_95_df$Algorithm <- "ACORN"
msvbase_wiki_15_4M_95_df$Algorithm <- "VBase"

wiki_data <- rbind(bfann_wiki_15_4M_95_df,
                   navix_wiki_15_4M_95_df, 
                   acorn_wiki_15_4M_95_df,
                   msvbase_wiki_15_4M_95_df
)

wiki_data$Dataset <- "Wiki15.4M"

# ------------- low sel





# 





youtube_data$Dataset <- "YoutubeRGB1M"
sift10m_data$Dataset <- "SIFT10M"
redcaps_data$Dataset <- "Redcaps4M"

# data <- redcaps_data
data <- rbind(youtube_data, sift10m_data, redcaps_data, wiki_data)

data$Dataset <- factor(
  data$Dataset,
  levels = c(
    "Redcaps4M",
    "SIFT10M",
    "YoutubeRGB1M",
    "Wiki15.4M"
  )
)
data$Algorithm <- factor(
  data$Algorithm,
  levels = c("EMA", "EMA_90","iRangeGraph", "NaviX","NaviX_90","ACORN", "Milvus", "VBase", "VBase_90", "90% Recall")
)

theme_Publication <- function(base_size = 8.5, base_family = "serif") {
  library(grid)
  library(ggthemes)
  theme_foundation(base_size = base_size, base_family = base_family) +
    theme(
      text = element_text(colour = "black"),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.35),
      axis.title = element_text(face = "bold", size = 9.5),
      axis.title.x = element_text(margin = margin(t = 2)),
      axis.title.y = element_text(margin = margin(r = 2)),
      axis.text = element_text(size = 7.8, colour = "black"),
      axis.line = element_blank(),
      axis.ticks = element_line(colour = "black", linewidth = 0.3),
      axis.ticks.length = unit(0.08, "cm"),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.text = element_text(size = 8.4),
      legend.key = element_blank(),
      legend.spacing.x = unit(0.08, "cm"),
      legend.margin = margin(0, 0, 1, 0),
      panel.grid.major = element_line(colour = "#D9D9D9", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      plot.margin = margin(2, 2, 1, 1),
      strip.background = element_rect(colour = NA, fill = "white"),
      strip.text = element_text(face = "bold", size = 9, margin = margin(b = 1))
    )
}

log10_minor_break = function(...) {
  function(x) {
    minx = floor(min(log10(x), na.rm = T)) - 1;
    maxx = ceiling(max(log10(x), na.rm = T)) + 1;
    n_major = maxx-minx+1;
    major_breaks = seq(minx, maxx, by = 1)
    minor_breaks = 
      rep(log10(seq(1, 9, by = 1)), times = n_major) + 
      rep(major_breaks, each = 9)
    return(10^(floor(minor_breaks)))
  }
}

color_mapping <- c(
  "EMA" = "#B72230",
  "NaviX" = "#104680",
  "NaviX_90" = "#104680",
  "Milvus" = "#317CB7",
  "VBase" = "#B6D7E8",
  "iRangeGraph" = "#7F7F7F",
  "ACORN" = "#6DADD1",
  "EMA_90" = "#B72230",
  "90% Recall" = "grey50",
  "VBase_90" = "#B6D7E8"
)

name_mapping <- c(
  "Milvus" = "Milvus(MT)",
  "VBase" = "VBase",
  "ACORN" = "ACORN",
  "NaviX" = "NaviX",
  "EMA" = "EMA",
  "iRangeGraph" = "iRangeGraph",
  "NaviX_90" = "NaviX 90\\%",
  "EMA_90" = "EMA 90\\%",
  "90% Recall" = "90\\% Recall",
  "VBase_90" = "VBase_90" 
)

shape_mapping <- c(
  "EMA" = 16, 
  "NaviX" = 17, 
  "ACORN" = 18,
  "Milvus" = 26,
  "VBase" = 15,
  "iRangeGraph" = 25, 
  "DiskANN" = 21,
  "90% Recall" = NA,
  "NaviX_90" = 16,
  "EMA_90" = 16, 
  "VBase_90" = 15
)

# 定义 线形
line_mapping <- c(
  "Milvus" = "solid", 
  "VBase" = "solid", 
  "ACORN" = "solid", 
  "NaviX" = "solid",
  "EMA" = "solid",
  "iRangeGraph" = "solid",
  "NaviX_90" = "dashed",
  "EMA_90" = "dashed",
  "90% Recall" = "dashed",
  "VBase_90" = "dashed"
)

GeomPaperPoint <- ggproto(
  "GeomPaperPoint",
  GeomPoint,
  draw_panel = function(data, panel_params, coord, na.rm = FALSE) {
    coords <- coord$transform(data, panel_params)
    grobs <- lapply(seq_len(nrow(coords)), function(i) {
      if (coords$shape[i] == 26) {
        angles <- seq(0, 2 * pi, length.out = 7) + pi / 6
        diameter <- coords$size[i] * 0.78
        grobTree(
          polygonGrob(
            x = 0.5 + 0.46 * cos(angles),
            y = 0.5 + 0.46 * sin(angles),
            gp = gpar(
              col = coords$colour[i],
              fill = coords$colour[i],
              lwd = coords$stroke[i] * .pt
            )
          ),
          vp = viewport(
            x = unit(coords$x[i], "native"),
            y = unit(coords$y[i], "native"),
            width = unit(diameter, "mm"),
            height = unit(diameter, "mm")
          )
        )
      } else {
        pointsGrob(
          x = unit(coords$x[i], "native"),
          y = unit(coords$y[i], "native"),
          pch = coords$shape[i],
          gp = gpar(
            col = coords$colour[i],
            fill = coords$fill[i],
            fontsize = coords$size[i] * .pt,
            lwd = coords$stroke[i] * .pt
          )
        )
      }
    })
    grobTree(children = do.call(gList, grobs))
  },
  draw_key = function(data, params, size) {
    if (isTRUE(data$shape == 26)) {
      angles <- seq(0, 2 * pi, length.out = 7) + pi / 6
      grobTree(
        polygonGrob(
          x = 0.5 + 0.46 * cos(angles),
          y = 0.5 + 0.46 * sin(angles),
          gp = gpar(col = data$colour, fill = data$colour, lwd = 0.5)
        ),
        vp = viewport(
          x = 0.5,
          y = 0.5,
          width = unit(2.2, "mm"),
          height = unit(2.2, "mm")
        )
      )
    } else {
      draw_key_point(data, params, size)
    }
  }
)

geom_paper_point <- function(mapping = NULL, data = NULL, ..., na.rm = FALSE,
                             show.legend = NA, inherit.aes = TRUE) {
  layer(
    geom = GeomPaperPoint,
    mapping = mapping,
    data = data,
    stat = "identity",
    position = "identity",
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(na.rm = na.rm, ...)
  )
}

guide_fun <- function(){
  guide_legend(
    nrow = 1,
    byrow = TRUE,
    position = "top",
    direction = "horizontal",
    title = NULL,
    keywidth = unit(0.65, "cm"),
    keyheight = unit(0.28, "cm"),
    override.aes = list(linewidth = 0.65, size = 2)
  )
}

# 自定义科学计数法标签格式的函数
scientific_format <- function() {
  function(x) {
    mantissa <- signif(x / 10^floor(log10(x)), 2)
    exponent <- floor(log10(x))
    if (mantissa == 1) {
      return(bquote(10^.(exponent)))
    } else {
      return(bquote(.(mantissa) * 10^.(exponent)))
    }
  }
}

source("paper_plot_style.R")
theme_Publication <- paper_theme
guide_fun <- paper_guide
color_mapping[names(color_mapping)] <- paper_colours[names(color_mapping)]
shape_mapping[names(shape_mapping)] <- paper_shapes[names(shape_mapping)]

plot_func <- function(scal) {
  legend_breaks <- c(
    "EMA", "NaviX", "ACORN", "VBase", "Milvus", "90% Recall"
  )

  ggplot(
    scal,
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
    geom_line(linewidth = 0.65) +
    geom_paper_point(size = 2.0, stroke = 0.45) +
    scale_shape_manual(
      values = shape_mapping,
      labels = name_mapping[legend_breaks],
      breaks = legend_breaks
    ) +
    scale_colour_manual(
      values = color_mapping,
      labels = name_mapping[legend_breaks],
      breaks = legend_breaks
    ) +
    scale_linetype_manual(
      values = line_mapping,
      labels = name_mapping[legend_breaks],
      breaks = legend_breaks
    ) +
    ggh4x::facetted_pos_scales(
      y = list(
        Dataset == "Wiki15.4M" ~ scale_y_continuous(
          limits = c(0, 1600),
          breaks = seq(0, 1600, by = 400),
          expand = expansion(mult = c(0.05, 0.02))
        ),
        Dataset == "SIFT10M" ~ scale_y_continuous(
          limits = c(0, 2100),
          breaks = seq(0, 2100, by = 600),
          expand = expansion(mult = c(0.05, 0.02))
        ),
        Dataset == "YoutubeRGB1M" ~ scale_y_continuous(
          limits = c(0, 650),
          breaks = seq(0, 600, by = 200),
          expand = expansion(mult = c(0.05, 0.02))
        ),
        Dataset == "Redcaps4M" ~ scale_y_continuous(
          limits = c(0, 3600),
          breaks = seq(0, 3600, by = 1200),
          expand = expansion(mult = c(0.05, 0.02))
        )
      )
    ) +
    scale_x_continuous(
      labels = function(x) format(x * 100, trim = TRUE),
      breaks = c(0.1, 0.4, 0.6, 0.8, 1)
    ) +
    labs(y = "QPS", x = "Selectivity (\\%)") +
    theme_Publication() +
    paper_compact_strip_theme() +
    theme(
      panel.grid.major.x = element_blank(),
      axis.title.y = element_text(angle = 90, vjust = 0.5)
    ) +
    guides(
      shape = guide_fun(nrow = 2, key_width = 0.42),
      colour = guide_fun(nrow = 2, key_width = 0.42),
      linetype = guide_fun(nrow = 2, key_width = 0.42)
    )
}



multi_plot <- function(input_data, output_file) {
  final_plot <- plot_func(input_data)

  tikz(
    file = output_file,
    width = 4.8,
    height = 1.65,
    standAlone = FALSE
  )
  print(final_plot)
  while (dev.cur() > 1) dev.off()
  cat("TikZ figure saved to ", output_file, "\n", sep = "")
}

output_file <- ("range_label_high_sel_qps_tikz.tex")
if (Sys.getenv("EMA_SKIP_PLOT") != "1") {
  multi_plot(data, output_file)
}

# input_file <- c("csv/sel_cpq_pruning_SeRF100000label_0.9recall_random_sift1M.csv",
#                 "csv/sel_cpq_pruning_SeRF100000label_0.9recall_real_redcaps1m.csv")
# output_file <- ("./plot/sel_cpq_serf_pruning.pdf")
# multi_plot(input_file, output_file, cpq_list)
