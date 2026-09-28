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

# ------------------- sift10m -----------------------


# bfann results
# Target recall: 0.95
bfann_sift10m_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(1348.73, 1278.78, 1128, 905, 684, 391)
)


# navix results
# Target recall: 0.95
navix_sift10m_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(148.30, 139.80, 101, 87.06, 44.38, 29.74)
)


# acorn results
# Target recall: 0.95
acorn_sift10m_95_df <- data.frame(
  x = c(0.10),
  y = c(36.45)
)


# milvus results
# Target recall: 0.95
milvus_sift10m_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(13.99, 13.36, 13.75, 14.69, 16.05, 16.39)
)

# VBase results
# Target recall: 0.9
VBase_sift10m_90_df <- data.frame(
  x = c(0.03, 0.02, 0.01),
  y = c(461.74, 321.81, 0.18)
)

bfann_sift10m_95_df$Algorithm <- "EMA"
navix_sift10m_95_df$Algorithm <- "NaviX"
acorn_sift10m_95_df$Algorithm <- "ACORN"
milvus_sift10m_95_df$Algorithm <- "Milvus"
VBase_sift10m_90_df$Algorithm <- "VBase_90"

sift10m_data <- rbind(bfann_sift10m_95_df, 
                      navix_sift10m_95_df,
                      acorn_sift10m_95_df,
                      milvus_sift10m_95_df
)

# ----------------------------- wiki ---------------------

# navix results
# Target recall: 0.95
navix_wiki_15_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(39.05, 31.55, 53.78, 37.32, 59.93, 12.54)
)


# acorn results
# Target recall: 0.95
acorn_wiki_15_4M_95_df <- data.frame(
  x = c(0.10),
  y = c(29.75)
)


# bfann results
# Target recall: 0.95
bfann_wiki_15_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(814, 692, 585, 386.12, 254.20, 161.05)
)


# VBase results
# Target recall: 0.95
VBase_wiki_15_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(410.43, 380.52, 379.07, 248.85, 180.26, 92.04)
)

bfann_wiki_15_4M_95_df$Algorithm <- "EMA"
navix_wiki_15_4M_95_df$Algorithm <- "NaviX"
VBase_wiki_15_4M_95_df$Algorithm <- "VBase"
acorn_wiki_15_4M_95_df$Algorithm <- "ACORN"

wiki_data <- rbind(navix_wiki_15_4M_95_df,  
                   acorn_wiki_15_4M_95_df,
                   bfann_wiki_15_4M_95_df,
                   VBase_wiki_15_4M_95_df
)


# -------------------- redcaps ------------------------

# bfann results
# Target recall: 0.95
bfann_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(1577.67, 1467, 1126.12, 808, 628, 334)
)


# navix results
# Target recall: 0.95
navix_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(425.49, 408.58, 335.80, 230.42, 191.47, 56.98)
)


# milvus results
# Target recall: 0.95
milvus_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(19, 20, 22.70, 25.17, 28.53, 30)
)

# VBase results
# Target recall: 0.95
VBase_Redcaps_4M_95_df <- data.frame(
  x = c(0.05, 0.03, 0.02, 0.01),
  y = c(251.79, 225.27, 48.23, 0.11)
)

# acorn results
# Target recall: 0.9
acorn_Redcaps_4M_90_df <- data.frame(
  x = c(0.10),
  y = c(51.15)
)

bfann_Redcaps_4M_95_df$Algorithm <- "EMA"
navix_Redcaps_4M_95_df$Algorithm <- "NaviX"
milvus_Redcaps_4M_95_df$Algorithm <- "Milvus"
VBase_Redcaps_4M_95_df$Algorithm <- "VBase"
acorn_Redcaps_4M_90_df$Algorithm <- "ACORN_90"

redcaps_data <- rbind(bfann_Redcaps_4M_95_df, 
                      navix_Redcaps_4M_95_df,
                      milvus_Redcaps_4M_95_df,
                      VBase_Redcaps_4M_95_df,
                      acorn_Redcaps_4M_90_df
)


#----------------------- youtube --------------

# bfann results
# Target recall: 0.95
bfann_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(349.89, 321.46, 293.15, 251, 170, 89)
)


# navix results
# Target recall: 0.95
navix_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(28.73, 29.34, 34.23, 21.87, 18.60, 10.21)
)


# milvus results
# Target recall: 0.95
milvus_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(57.59, 57.41, 56.97, 62.48, 72.10, 74.43)
)

bfann_youtube_rgb_95_df$Algorithm <- "EMA"
navix_youtube_rgb_95_df$Algorithm <- "NaviX"
milvus_youtube_rgb_95_df$Algorithm <- "Milvus"


youtube_data <- rbind(bfann_youtube_rgb_95_df, 
                      navix_youtube_rgb_95_df,
                      milvus_youtube_rgb_95_df
)
# -------------------------------------------




legend_90_df <- data.frame(
  x = NA,
  y = NA,
  Algorithm = "90% Recall"
)


legend_90_df$Algorithm <- "90% Recall"







youtube_data$Dataset <- "YoutubeRGB1M"
sift10m_data$Dataset <- "SIFT10M"
wiki_data$Dataset <- "Wiki15.4M"
redcaps_data$Dataset <- "Redcaps4M"

# data <- redcaps_data
data <- rbind(youtube_data, sift10m_data, wiki_data, redcaps_data)

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
  levels = c("EMA", "EMA_90","iRangeGraph", "NaviX","NaviX_90","ACORN", "ACORN_90", "VBase", "VBase_90", "Milvus", "90% Recall")
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
  "VBase" = "#6DADD1",
  "iRangeGraph" = "#B6D7E8",
  "ACORN" = "#7F7F7F",
  "EMA_90" = "#B72230",
  "90% Recall" = "grey50",
  "VBase_90" = "#6DADD1"
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
  "Milvus" = 15,  
  "VBase" = 23, 
  "iRangeGraph" = 25, 
  "DiskANN" = 21,
  "90% Recall" = NA,
  "NaviX_90" = 16,
  "EMA_90" = 16, 
  "VBase_90" = 23
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
  "VBase_90" = "dashed", 
  "ACORN_90" = "dashed", 
  "90% Recall" = "dashed"
)

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
    scale_shape_manual(values = shape_mapping) +
    scale_colour_manual(values = color_mapping) +
    scale_linetype_manual(values = line_mapping) +
    ggh4x::facetted_pos_scales(
      y = list(
        Dataset == "Wiki15.4M" ~ scale_y_continuous(
          limits = c(0, 900),
          breaks = seq(0, 800, by = 200),
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
        Dataset == "Redcaps4M" ~ scale_y_continuous(
          limits = c(0, 1800),
          breaks = seq(0, 1800, by = 600),
          expand = expansion(mult = c(0.05, 0.02))
        )
      )
    ) +
    scale_x_log10(
      labels = function(x) format(x * 100, trim = TRUE),
      breaks = c(0.01, 0.02, 0.05, 0.1)
    ) +
    labs(y = "QPS", x = "Selectivity (\\%)") +
    theme_Publication() +
    paper_compact_strip_theme() +
    theme(
      legend.position = "none",
      panel.grid.major.x = element_blank(),
      axis.title.y = element_text(angle = 90, vjust = 0.5)
    )
}



multi_plot <- function(input_data, output_file) {
  
  
  final_plot <- plot_func(input_data)
  
  tikz(file = output_file, width = 4.8, height = 1.38, standAlone = FALSE)
  print(final_plot)
  while (dev.cur() > 1) dev.off()
  
  
  cat("png已成功保存在")
  cat(output_file)
  cat("\n")
}




output_file <- ("range_label_sel_qps_tikz.tex")
if (Sys.getenv("EMA_SKIP_PLOT") != "1") {
  multi_plot(data, output_file)
}

# input_file <- c("csv/sel_cpq_pruning_SeRF100000label_0.9recall_random_sift1M.csv",
#                 "csv/sel_cpq_pruning_SeRF100000label_0.9recall_real_redcaps1m.csv")
# output_file <- ("./plot/sel_cpq_serf_pruning.pdf")
# multi_plot(input_file, output_file, cpq_list)
