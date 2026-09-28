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


# ---------------------- redcaps ---------------------

# navix results
# Target recall: 0.95
navix_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(220.27, 197.32, 189.87, 152.33, 170.83, 69.25)
)

# bfann results
# Target recall: 0.95
bfann_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(758.13, 747.76, 717.20, 534.85, 456.27, 302.52)
)

# milvus results
# Target recall: 0.95
milvus_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(11.51, 10.27, 10.01, 10.42, 10.65, 11.11)
)

# acorn results
# Target recall: 0.8
acorn_Redcaps_4M_80_df <- data.frame(
  x = c(0.10, 0.07),
  y = c(171.65, 94.16)
)

# msvbase results
# Target recall: 0.95
msvbase_Redcaps_4M_95_df <- data.frame(
  x = c(0.05, 0.03, 0.02, 0.01),
  y = c(373.96, 255.33, 70.31, 0.76)
)

bfann_Redcaps_4M_95_df$Algorithm <- "EMA"
navix_Redcaps_4M_95_df$Algorithm <- "NaviX"
msvbase_Redcaps_4M_95_df$Algorithm <- "VBase"
acorn_Redcaps_4M_80_df$Algorithm <- "ACORN_80"
milvus_Redcaps_4M_95_df$Algorithm <- "Milvus"

redcaps_data <- rbind(bfann_Redcaps_4M_95_df, 
                      navix_Redcaps_4M_95_df,
                      msvbase_Redcaps_4M_95_df,
                      acorn_Redcaps_4M_80_df,
                      milvus_Redcaps_4M_95_df
)

# ------------------- sift10m -----------------------

# bfann results
# Target recall: 0.95
bfann_sift10m_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(912.31, 830.54, 500.07, 393.57, 476.28, 342.06)
)


# navix results
# Target recall: 0.95
navix_sift10m_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(73.35, 105.01, 53.96, 72.28, 56.34, 17.55)
)


# acorn results
# Target recall: 0.95
acorn_sift10m_95_df <- data.frame(
  x = c(0.10),
  y = c(27.09)
)


# milvus results
# Target recall: 0.95
milvus_sift10m_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(6.41, 6.15, 5.71, 5.99, 6.08, 6.28)
)


# VBase results
# Target recall: 0.95
VBase_sift10m_95_df <- data.frame(
  x = c(0.01),
  y = c(0.58)
)

# diskann results
# Target recall: 0.83
diskann_sift10m_80_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(301.21, 98.34, 145.46, 160.34, 52.82, 41.56)
)


legend_90_df <- data.frame(
  x = NA,
  y = NA,
  Algorithm = "90% Recall"
)
legend_80_df <- data.frame(
  x = NA,
  y = NA,
  Algorithm = "80% Recall"
)



legend_90_df$Algorithm <- "90% Recall"
legend_80_df$Algorithm <- "80% Recall"


bfann_sift10m_95_df$Algorithm <- "EMA"
navix_sift10m_95_df$Algorithm <- "NaviX"
acorn_sift10m_95_df$Algorithm <- "ACORN"
milvus_sift10m_95_df$Algorithm <- "Milvus"
VBase_sift10m_95_df$Algorithm <- "VBase"
diskann_sift10m_80_df$Algorithm <- "DiskANN_80"

sift10m_data <- rbind(bfann_sift10m_95_df, 
                      navix_sift10m_95_df,
                      acorn_sift10m_95_df,
                      VBase_sift10m_95_df,
                      milvus_sift10m_95_df,
                      diskann_sift10m_80_df,
                      legend_80_df
)



# ----------------------------- wiki ---------------------
# bfann results
# Target recall: 0.95
bfann_wiki_15_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(377.81, 355.05, 300.96, 307.88, 287.47, 233.93)
)


# navix results
# Target recall: 0.95
navix_wiki_15_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(99.69, 89.14, 103.37, 79.07, 52.92, 28.23)
)


# acorn results
# Target recall: 0.95
acorn_wiki_15_4M_95_df <- data.frame(
  x = c(0.10, 0.07),
  y = c(40.04, 9.47)
)


# diskann results
# Target recall: 0.95
diskann_wiki_15_4M_95_df <- data.frame(
  x = c(0.10),
  y = c(240.38)
)

# diskann results
# Target recall: 0.83
diskann_wiki_15_4M_80_df <- data.frame(
  x = c(0.10, 0.07, 0.05),
  y = c(802.49, 211.78, 50.06)
)


# VBase results
# Target recall: 0.95
VBase_wiki_15_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(7.63, 10.65, 9.67, 8.47, 6.97, 2.22)
)

bfann_wiki_15_4M_95_df$Algorithm <- "EMA"
navix_wiki_15_4M_95_df$Algorithm <- "NaviX"
VBase_wiki_15_4M_95_df$Algorithm <- "VBase"
acorn_wiki_15_4M_95_df$Algorithm <- "ACORN"
diskann_wiki_15_4M_95_df$Algorithm <- "DiskANN"
diskann_wiki_15_4M_80_df$Algorithm <- "DiskANN_80"

wiki_data <- rbind(navix_wiki_15_4M_95_df,  
                   acorn_wiki_15_4M_95_df,
                   bfann_wiki_15_4M_95_df,
                   VBase_wiki_15_4M_95_df,
                   diskann_wiki_15_4M_95_df,
                   diskann_wiki_15_4M_80_df
)


# -------------------- redcaps ------------------------


# bfann_Redcaps_4M_95_df$Algorithm <- "EMA"
# navix_Redcaps_4M_95_df$Algorithm <- "NaviX"
# milvus_Redcaps_4M_95_df$Algorithm <- "Milvus"
# VBase_Redcaps_4M_95_df$Algorithm <- "VBase"
# acorn_Redcaps_4M_90_df$Algorithm <- "ACORN_90"
# 
# redcaps_data <- rbind(bfann_Redcaps_4M_95_df, 
#                       navix_Redcaps_4M_95_df,
#                       milvus_Redcaps_4M_95_df,
#                       VBase_Redcaps_4M_95_df,
#                       acorn_Redcaps_4M_90_df,
#                       legend_80_df
# )


#----------------------- youtube --------------

# bfann results
# Target recall: 0.95
bfann_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(358.53, 285.63, 235.21, 160.15, 125.91, 72.59)
)

# navix results
# Target recall: 0.95
navix_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(71.55, 36.58, 35.13, 26.04, 20.04, 9.06)
)

# milvus results
# Target recall: 0.95
milvus_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(16.04, 14.70, 12.61, 13.45, 13.80, 14.77)
)

# diskann results
# Target recall: 0.95
diskann_youtube_rgb_95_df <- data.frame(
  x = c(NA),
  y = c(NA)
)

# diskann results
# Target recall: 0.8
diskann_youtube_rgb_80_df <- data.frame(
  x = c(0.10, 0.07),
  y = c(337.74, 41.80)
)

bfann_youtube_rgb_95_df$Algorithm <- "EMA"
navix_youtube_rgb_95_df$Algorithm <- "NaviX"
milvus_youtube_rgb_95_df$Algorithm <- "Milvus"
diskann_youtube_rgb_95_df$Algorithm <- "DiskANN"
diskann_youtube_rgb_80_df$Algorithm <- "DiskANN_80"

youtube_data <- rbind(bfann_youtube_rgb_95_df, 
                      navix_youtube_rgb_95_df,
                      milvus_youtube_rgb_95_df,
                      diskann_youtube_rgb_80_df,
                      diskann_youtube_rgb_95_df
)
# -------------------------------------------











youtube_data$Dataset <- "YoutubeRGB1M"
sift10m_data$Dataset <- "SIFT10M"
wiki_data$Dataset <- "Wiki15.4M"
redcaps_data$Dataset <- "Redcaps4M"

# data <- redcaps_data
data <- rbind(redcaps_data, youtube_data)

data$Algorithm <- factor(
  data$Algorithm,
  levels = c("EMA", "EMA_90","iRangeGraph", 
             "NaviX","NaviX_90","ACORN", "ACORN_90", 
             "VBase", "VBase_90", "Milvus", "DiskANN", 
             "DiskANN_80", "90% Recall", "80% Recall")
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
            plot.background = element_rect(fill = "white", colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold",size = rel(2.5)),
            axis.title.x = element_text(vjust = -0.2),
            axis.text = element_text(size = rel(2)), 
            axis.line = element_line(colour="black"),
            axis.ticks = element_line(),
            
            axis.ticks.y = element_line(size=1),
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
  "NaviX_90" = "#104680",
  "Milvus" = "#317CB7",
  "VBase" = "#6DADD1",
  "iRangeGraph" = "#B6D7E8",
  "ACORN" = "#7F7F7F",
  "ACORN_90" = "#7F7F7F",
  "EMA_90" = "#B72230",
  "90% Recall" = "grey50",
  "80% Recall" = "grey50",
  "VBase_90" = "#6DADD1",
  "DiskANN" = "#4D4D4D",
  "DiskANN_80" = "#4D4D4D"
)

name_mapping <- c(
  "Milvus" = "Milvus(MT)",
  "VBase" = "VBase",
  "VBase_90" = "VBase 90\\%",
  "ACORN" = "ACORN",
  "ACORN_90" = "ACORN_90",
  "NaviX" = "NaviX",
  "EMA" = "EMA",
  "iRangeGraph" = "iRangeGraph",
  "DiskANN" = "Filtered DiskANN",
  "DiskANN_80" = "Filtered DiskANN 80\\%",
  "NaviX_90" = "NaviX 90\\%",
  "EMA_90" = "EMA 90\\%",
  "90% Recall" = "90\\% Recall",
  "80% Recall" = "80\\% Recall"
)

# 定义 12 种不同的形状
# shape_mapping <- c(
#   "Milvus" = 8,  # 星号
#   "VBase" = 17, # 实心三角形
#   "ACORN" = 16, # 实心圆
#   "NaviX" = 15, # 实心正方形
#   "EMA" = 18, # 实心菱形
#   "iRangeGraph" = 4  # 叉号
# )


shape_mapping <- c(
  "EMA" = 16, 
  "NaviX" = 17, 
  "ACORN" = 18, 
  "Milvus" = 15,  
  "VBase" = 23, 
  "iRangeGraph" = 25, # 不在同一处出现
  "DiskANN" = 25,
  "DiskANN_80" = 25,
  "90% Recall" = NA,
  "NaviX_90" = 16,
  "EMA_90" = 16, 
  "VBase_90" = 23,
  "90% Recall" = NA,
  "80% Recall" = NA
)

# 定义 线形
line_mapping <- c(
  "Milvus" = "solid", 
  "VBase" = "solid", 
  "ACORN" = "solid", 
  "NaviX" = "solid",
  "EMA" = "solid",
  "iRangeGraph" = "solid",
  "DiskANN" = "solid",
  "DiskANN_80" = "dotted",
  "NaviX_90" = "dashed",
  "EMA_90" = "dashed",
  "VBase_90" = "dashed", 
  "ACORN_90" = "dashed", 
  "90% Recall" = "dashed", 
  "80% Recall" = "dotted"
)

guide_fun <- function(){
  guide_legend(
    nrow = 2,  # 设置图例为 1 行
    byrow = TRUE,  # 按行填充图例项
    position = "top",
    direction ="horizontal",
    title = NULL,
    keywidth = 6, 
    keyheight = 2
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
guide_fun <- function() paper_guide(nrow = 2, key_width = 0.45)
color_mapping[names(color_mapping)] <- paper_colours[names(color_mapping)]
shape_mapping[names(shape_mapping)] <- paper_shapes[names(shape_mapping)]

plot_func <- function(scal) {
  
  p_scal <- ggplot(
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
    # ✅ 用 facet_grid2：每个面板都有自己的 y 轴，并允许 y 独立
    ggh4x::facet_grid2(
      . ~ Dataset,
      scales = "free_y",      # ← 必须加这个
      axes = "all",
      independent = "y"
    ) +
    
    geom_line(linewidth = 0.65) +
    geom_paper_point(size = 2.1, stroke = 0.45) +
    
    scale_shape_manual(
      values = shape_mapping,
      labels = name_mapping[c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "DiskANN", "90% Recall", "80% Recall")],
      breaks = c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "DiskANN", "90% Recall", "80% Recall")
    ) +
    scale_colour_manual(
      values = color_mapping,
      labels = name_mapping[c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "DiskANN", "90% Recall", "80% Recall")],
      breaks = c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "DiskANN", "90% Recall", "80% Recall")
    ) +
    scale_linetype_manual(
      values = line_mapping,
      labels = name_mapping[c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "DiskANN", "90% Recall", "80% Recall")],
      breaks = c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "DiskANN", "90% Recall", "80% Recall")
    ) +
    
    # ✅ 每个 Dataset 单独设置 y 轴（这段你写的是对的）
    ggh4x::facetted_pos_scales(
      y = list(
        Dataset == "YoutubeRGB1M" ~ scale_y_continuous(
          limits = c(0, 400),
          breaks = seq(0, 400, by = 100),
          expand = expansion(mult = c(0.05, 0.02))
        ),
        Dataset == "Redcaps4M" ~ scale_y_continuous(
          limits = c(0, 800),
          breaks = seq(0, 800, by = 200),
          expand = expansion(mult = c(0.05, 0.02))
        )
      )
    ) +
    
    scale_x_log10(
      labels = function(x) format(x * 100, trim = TRUE),
      breaks = c(0.01, 0.02, 0.03, 0.05, 0.07, 0.1)
    ) +
    
    labs(y = "QPS", x = "Selectivity (\\%)") +
    theme_Publication() +
    theme(
      axis.ticks.x = element_line(),
      axis.ticks.y = element_line(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      plot.caption = element_text(hjust = 0.5, vjust = 0.5),
      axis.title.y = element_text(angle = 90, vjust = 0.5, hjust = 0.5)
    ) +
    guides(
      shape = guide_fun(),
      colour = guide_fun(),
      linetype = guide_fun()
    )
  
  return(p_scal)
}



multi_plot <- function(input_data, output_file) {
  
  
  final_plot <- plot_func(input_data)
  
  tikz(file = output_file, width = 3.35, height = 1.9, standAlone = FALSE)
  print(final_plot)
  while (dev.cur() > 1) dev.off()
  

  cat("png已成功保存在")
  cat(output_file)
  cat("\n")

}




output_file <- ("label_sel_qps_tikz.tex")
if (Sys.getenv("EMA_SKIP_PLOT") != "1") {
  multi_plot(data, output_file)
}

# input_file <- c("csv/sel_cpq_pruning_SeRF100000label_0.9recall_random_sift1M.csv",
#                 "csv/sel_cpq_pruning_SeRF100000label_0.9recall_real_redcaps1m.csv")
# output_file <- ("./plot/sel_cpq_serf_pruning.pdf")
# multi_plot(input_file, output_file, cpq_list)
