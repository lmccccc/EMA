library(readxl)
library(scales)
library(tikzDevice)
library(patchwork)
library(reshape2)
library(plyr)
library(latex2exp)
library(tidyr)
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

# =====================================================================
# Composite predicate (range + label) on 4 datasets.
# Selectivity sweep: 1% -- 10%
# Only Redcaps4M has measured data for now; the other three datasets are
# placeholders to be filled in later.
# =====================================================================


# -------------------- redcaps ------------------------

# bfann (EMA) results -- composite predicate
# Target recall: 0.95
bfann_Redcaps_4M_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(410, 651, 902, 1089, 1343, 1494)
)

# acorn results -- only reached recall target at 10%
# Target recall: 0.95
acorn_Redcaps_4M_95_df <- data.frame(
  x = c(0.10),
  y = c(6.82)
)

# acorn @ 0.90 recall
acorn_Redcaps_4M_90_df <- data.frame(
  x = c(0.10),
  y = c(56.3)
)

bfann_Redcaps_4M_95_df$Algorithm <- "EMA"
acorn_Redcaps_4M_95_df$Algorithm <- "ACORN"
acorn_Redcaps_4M_90_df$Algorithm <- "ACORN_90"

# navix results -- composite predicate
# Target recall: 0.95
navix_Redcaps_4M_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(43, 128, 173, 246, 282, 311)
)
navix_Redcaps_4M_95_df$Algorithm <- "NaviX"

# milvus results (multi-threaded; cannot be disabled)
# Target recall: 0.95
milvus_Redcaps_4M_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(12.8, 12.7, 13.3, 13.2, 14.2, 13.8)
)
milvus_Redcaps_4M_95_df$Algorithm <- "Milvus"

# vbase results
# Target recall: 0.95 for 1%-5%
VBase_Redcaps_4M_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05),
  y = c(0.21, 196.4, 251.7, 447.8)
)
# Target recall: 0.90 for 7%-10% (did not reach 95% recall)
VBase_Redcaps_4M_90_df <- data.frame(
  x = c(0.07, 0.10),
  y = c(557.5, 609.2)
)
VBase_Redcaps_4M_95_df$Algorithm <- "VBase"
VBase_Redcaps_4M_90_df$Algorithm <- "VBase_90"

redcaps_data <- rbind(bfann_Redcaps_4M_95_df,
                      acorn_Redcaps_4M_95_df,
                      acorn_Redcaps_4M_90_df,
                      navix_Redcaps_4M_95_df,
                      milvus_Redcaps_4M_95_df,
                      VBase_Redcaps_4M_95_df,
                      VBase_Redcaps_4M_90_df)


# -------------------- sift10m ------------------------
bfann_sift10m_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(479, 584, 827, 1004, 1290, 1496)
)
# acorn @ 0.90 recall
acorn_sift10m_90_df <- data.frame(
  x = c(0.07, 0.10),
  y = c(36.7, 67.4)
)
# navix @ 0.95 recall
navix_sift10m_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(17, 46, 61, 70, 102, 101)
)
bfann_sift10m_95_df$Algorithm <- "EMA"
acorn_sift10m_90_df$Algorithm <- "ACORN_90"
navix_sift10m_95_df$Algorithm <- "NaviX"
# milvus results (multi-threaded; cannot be disabled)
# Target recall: 0.95
milvus_sift10m_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(8.5, 8.4, 9.0, 9.2, 10.0, 9.3)
)
milvus_sift10m_95_df$Algorithm <- "Milvus"
# vbase results
# Target recall: 0.95 for 1%-3% (no data for higher selectivity)
VBase_sift10m_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03),
  y = c(0.32, 343.6, 448.2)
)
VBase_sift10m_95_df$Algorithm <- "VBase"
sift10m_data <- rbind(bfann_sift10m_95_df,
                      acorn_sift10m_90_df,
                      navix_sift10m_95_df,
                      milvus_sift10m_95_df,
                      VBase_sift10m_95_df)


# ----------------------------- wiki ---------------------
bfann_wiki_15_4M_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(83.9, 286, 370, 396, 573, 986)
)
# acorn @ 0.90 recall
acorn_wiki_15_4M_90_df <- data.frame(
  x = c(0.07, 0.10),
  y = c(34.4, 62.8)
)
# navix @ 0.95 recall
navix_wiki_15_4M_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(12, 58, 64, 96, 114, 141)
)
bfann_wiki_15_4M_95_df$Algorithm <- "EMA"
acorn_wiki_15_4M_90_df$Algorithm <- "ACORN_90"
navix_wiki_15_4M_95_df$Algorithm <- "NaviX"
# vbase results
# Target recall: 0.95
VBase_wiki_15_4M_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(1.95, 73.06, 7.29, 8.22, 9.38, 24.73)
)
VBase_wiki_15_4M_95_df$Algorithm <- "VBase"
wiki_data <- rbind(bfann_wiki_15_4M_95_df,
                   acorn_wiki_15_4M_90_df,
                   navix_wiki_15_4M_95_df,
                   VBase_wiki_15_4M_95_df)


# ----------------------- youtube --------------
# ACORN did not reach 0.90 recall on YoutubeRGB1M -- leave out.
bfann_youtube_rgb_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(42, 130, 175, 184, 214, 330)
)
bfann_youtube_rgb_95_df$Algorithm <- "EMA"
# navix @ 0.95 recall
navix_youtube_rgb_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(3.7, 18, 13, 27, 37, 49)
)
navix_youtube_rgb_95_df$Algorithm <- "NaviX"
# milvus results (multi-threaded; cannot be disabled)
# Target recall: 0.95
milvus_youtube_rgb_95_df <- data.frame(
  x = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.10),
  y = c(33.8, 30.8, 32.5, 31.0, 34.0, 28.2)
)
milvus_youtube_rgb_95_df$Algorithm <- "Milvus"
youtube_data <- rbind(bfann_youtube_rgb_95_df,
                      navix_youtube_rgb_95_df,
                      milvus_youtube_rgb_95_df)
# -------------------------------------------


youtube_data$Dataset <- "YoutubeRGB1M"
sift10m_data$Dataset <- "SIFT10M"
wiki_data$Dataset    <- "Wiki15.4M"
redcaps_data$Dataset <- "Redcaps4M"

data <- rbind(youtube_data, sift10m_data, wiki_data, redcaps_data)

# Add a phantom row so the legend includes a single "90% Recall" entry
# (grey dashed) that explains all _90 series on the plot.
legend_90_df <- data.frame(
  x = NA_real_,
  y = NA_real_,
  Algorithm = "90% Recall",
  Dataset = "Redcaps4M"
)
data <- rbind(data, legend_90_df)

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

theme_Publication <- function(base_size=14, base_family="serif") {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size=base_size, base_family=base_family)
    + theme(plot.title = element_text(face = "bold",
                                      size = rel(2), hjust = 0.5),
            plot.caption = element_text(face = "bold",
                                        size = rel(2), hjust = 0.5),
            text = element_text(),
            panel.background = element_rect(fill = "white", colour = NA),
            plot.background  = element_rect(fill = "white", colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold",size = rel(2.5)),
            axis.title.x = element_text(vjust = -0.2),
            axis.text = element_text(size = rel(2)),
            axis.line = element_line(colour="black"),
            axis.ticks = element_line(),

            axis.ticks.y = element_line(linewidth=1),
            axis.ticks.length.y = unit(0.3, "cm"),

            legend.text=element_text(size=rel(2.5)),
            panel.grid.major = element_line(colour="#C7C8C7", size=0.2),
            panel.grid.minor = element_line(colour="#f0f0f0", size=0),
            plot.margin=unit(c(5,3,3,3),"mm"),
            strip.background=element_rect(colour=NA, fill="white"),
            strip.text = element_text(face="bold", size=rel(2))
    ))

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
  "NaviX_90" = "grey50",
  "Milvus" = "#317CB7",
  "VBase" = "#6DADD1",
  "iRangeGraph" = "#B6D7E8",
  "ACORN" = "#7F7F7F",
  "ACORN_90" = "grey50",
  "EMA_90" = "grey50",
  "90% Recall" = "grey50",
  "VBase_90" = "grey50"
)

name_mapping <- c(
  "Milvus" = "Milvus(MT)",
  "VBase" = "VBase",
  "ACORN" = "ACORN",
  "ACORN_90" = "ACORN 90\\%",
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
  "ACORN_90" = 18,
  "Milvus" = 15,
  "VBase" = 23,
  "iRangeGraph" = 25,
  "DiskANN" = 21,
  "90% Recall" = NA,
  "NaviX_90" = 16,
  "EMA_90" = 16,
  "VBase_90" = 23
)

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
    direction ="horizontal",
    title = NULL,
    keywidth = 6,
    keyheight = 2
  )
}

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
          limits = c(0, 1100),
          breaks = seq(0, 1000, by = 250),
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
        Dataset == "Redcaps4M" ~ scale_y_continuous(
          limits = c(0, 1600),
          breaks = seq(0, 1600, by = 400),
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
      panel.grid.minor = element_blank(),
      axis.title.y = element_text(angle = 90, vjust = 0.5, hjust = 0.5)
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


output_file <- ("range_label_composite_sel_qps_tikz.tex")
if (Sys.getenv("EMA_SKIP_PLOT") != "1") {
  multi_plot(data, output_file)
}
