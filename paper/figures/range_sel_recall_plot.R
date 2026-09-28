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

# navix results
# Target recall: 0.95
navix_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(277.76, 276.06, 290.14, 220.44, 181.96, 62.23)
)


# irange results
# Target recall: 0.95
irange_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.01),
  y = c(3137.70, 3340.87, 3376.74, 3356.03, 3137.56)
)

# bfann results
# Target recall: 0.95
bfann_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(1518, 1250, 1067, 699.21, 423.78, 109.08)
)

# acorn results
# Target recall: 0.95
acorn_Redcaps_4M_95_df <- data.frame(
  x = c(0.10),
  y = c(30)
)


# milvus results
# Target recall: 0.95
milvus_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(152.94, 59.32, 60.00, 69.77, 93.77, 94.07)
)

# VBase results
# Target recall: 0.95
msvbase_Redcaps_4M_95_df <- data.frame(
  x = c(0.10, 0.05, 0.03, 0.02, 0.01),
  y = c(625, 427.01, 330.46, 246.80, 159.37)
)




# bfann results
# Target recall: 0.95
bfann_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02,0.01),
  y = c(436.77, 365.95, 290.93, 190.07, 130.96,9)
)

# bfann results
# Target recall: 0.9
bfann_youtube_rgb_90_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(693.73, 636.37, 551.18, 400.78, 316.94, 122.51)
)


# milvus results
# Target recall: 0.95
milvus_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(144.16, 105.27, 97.57, 121.25, 130.95, 175.18)
)


# irange results
# Target recall: 0.95
irange_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.01),
  y = c(905.02, 1150, 1261, 1474, 1516.06)
)


# navix results
# Target recall: 0.95
navix_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03),
  y = c(47.42, 42.85, 35.91, 29.53)
)

# navix results
# Target recall: 0.9
navix_youtube_rgb_90_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(214.14, 174.69, 154.60, 129.08, 84.36, 31.08)
)

legend_90_df <- data.frame(
  x = NA,
  y = NA,
  Algorithm = "90% Recall"
)


bfann_youtube_rgb_95_df$Algorithm <- "EMA"
navix_youtube_rgb_95_df$Algorithm <- "NaviX"
irange_youtube_rgb_95_df$Algorithm <- "iRangeGraph"
milvus_youtube_rgb_95_df$Algorithm <- "Milvus"
bfann_youtube_rgb_90_df$Algorithm <- "EMA_90"
navix_youtube_rgb_90_df$Algorithm <- "NaviX_90"
legend_90_df$Algorithm <- "90% Recall"

bfann_Redcaps_4M_95_df$Algorithm <- "EMA"
navix_Redcaps_4M_95_df$Algorithm <- "NaviX"
irange_Redcaps_4M_95_df$Algorithm <- "iRangeGraph"
milvus_Redcaps_4M_95_df$Algorithm <- "Milvus"
acorn_Redcaps_4M_95_df$Algorithm <- "ACORN"
msvbase_Redcaps_4M_95_df$Algorithm <- "VBase"

youtube_data <- rbind(bfann_youtube_rgb_95_df, 
                      irange_youtube_rgb_95_df, 
                      navix_youtube_rgb_95_df,  
                      milvus_youtube_rgb_95_df,
                      bfann_youtube_rgb_90_df,
                      navix_youtube_rgb_90_df,
                      legend_90_df
)
redcaps_data <- rbind(bfann_Redcaps_4M_95_df, 
                      irange_Redcaps_4M_95_df, 
                      navix_Redcaps_4M_95_df, 
                      milvus_Redcaps_4M_95_df,
                      acorn_Redcaps_4M_95_df,
                      msvbase_Redcaps_4M_95_df)

youtube_data$Dataset <- "YoutubeRGB1M"
# sift10m_data$Dataset <- "SIFT10M"
# wiki_data$Dataset <- "Wiki15.4M"
redcaps_data$Dataset <- "Redcaps4M"

# data <- redcaps_data
data <- rbind(youtube_data, redcaps_data)

data$Algorithm <- factor(
  data$Algorithm,
  levels = c("EMA", "EMA_90","iRangeGraph", "NaviX","NaviX_90","ACORN", "VBase", "Milvus", "90% Recall")
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
  "90% Recall" = "90\\% Recall"
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
  "90% Recall" = "dashed"
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
      labels = name_mapping[c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "90% Recall")],
      breaks = c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "90% Recall")
    ) +
    scale_colour_manual(
      values = color_mapping,
      labels = name_mapping[c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "90% Recall")],
      breaks = c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "90% Recall")
    ) +
    scale_linetype_manual(
      values = line_mapping,
      labels = name_mapping[c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "90% Recall")],
      breaks = c("EMA", "NaviX", "ACORN", "VBase", "Milvus", "iRangeGraph", "90% Recall")
    ) +
    
    # ✅ 每个 Dataset 单独设置 y 轴（这段你写的是对的）
    ggh4x::facetted_pos_scales(
      y = list(
        Dataset == "YoutubeRGB1M" ~ scale_y_continuous(
          limits = c(0, 1600),
          breaks = seq(0, 1600, by = 400),
          expand = expansion(mult = c(0.05, 0.02))
        ),
        Dataset == "Redcaps4M" ~ scale_y_continuous(
          limits = c(0, 3600),
          breaks = seq(0, 3600, by = 600),
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




output_file <- ("range_sel_qps_tikz.tex")
if (Sys.getenv("EMA_SKIP_PLOT") != "1") {
  multi_plot(data, output_file)
}

# input_file <- c("csv/sel_cpq_pruning_SeRF100000label_0.9recall_random_sift1M.csv",
#                 "csv/sel_cpq_pruning_SeRF100000label_0.9recall_real_redcaps1m.csv")
# output_file <- ("./plot/sel_cpq_serf_pruning.pdf")
# multi_plot(input_file, output_file, cpq_list)
