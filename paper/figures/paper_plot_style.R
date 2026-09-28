library(ggplot2)
library(ggthemes)
library(grid)

paper_colours <- c(
  "EMA" = "#B72230",
  "EMA_90" = "#B72230",
  "NaviX" = "#104680",
  "NaviX_90" = "#104680",
  "Milvus" = "#317CB7",
  "ACORN" = "#6DADD1",
  "ACORN_90" = "#6DADD1",
  "ACORN_80" = "#6DADD1",
  "VBase" = "#B6D7E8",
  "VBase_90" = "#B6D7E8",
  "VBase_80" = "#B6D7E8",
  "iRangeGraph" = "#6A3D9A",
  "DiskANN" = "#C66A1B",
  "DiskANN_80" = "#C66A1B",
  "90% Recall" = "#7F7F7F",
  "80% Recall" = "#7F7F7F"
)

paper_shapes <- c(
  "EMA" = 16,
  "EMA_90" = 16,
  "NaviX" = 17,
  "NaviX_90" = 17,
  "ACORN" = 18,
  "ACORN_90" = 18,
  "ACORN_80" = 18,
  "VBase" = 15,
  "VBase_90" = 15,
  "VBase_80" = 15,
  "Milvus" = 26,
  "iRangeGraph" = 27,
  "DiskANN" = 28,
  "DiskANN_80" = 28,
  "90% Recall" = NA,
  "80% Recall" = NA
)

paper_theme <- function(base_size = 8.5, base_family = "serif") {
  theme_foundation(base_size = base_size, base_family = base_family) +
    theme(
      text = element_text(colour = "black"),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.35),
      axis.title = element_text(face = "bold", size = base_size + 1),
      axis.title.x = element_text(margin = margin(t = 2)),
      axis.title.y = element_text(margin = margin(r = 2)),
      axis.text = element_text(size = base_size - 0.7, colour = "black"),
      axis.line = element_blank(),
      axis.ticks = element_line(colour = "black", linewidth = 0.3),
      axis.ticks.length = unit(0.08, "cm"),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.text = element_text(size = base_size - 0.1),
      legend.key = element_blank(),
      legend.spacing.x = unit(0.08, "cm"),
      legend.margin = margin(0, 0, 1, 0),
      panel.grid.major = element_line(colour = "#D9D9D9", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      plot.margin = margin(2, 2, 1, 1),
      strip.background = element_rect(colour = NA, fill = "white"),
      strip.text = element_text(
        face = "bold",
        size = base_size + 0.5,
        margin = margin(b = 1)
      )
    )
}

paper_guide <- function(nrow = 1, key_width = 0.65) {
  guide_legend(
    nrow = nrow,
    byrow = TRUE,
    position = "top",
    direction = "horizontal",
    title = NULL,
    keywidth = unit(key_width, "cm"),
    keyheight = unit(0.28, "cm"),
    override.aes = list(linewidth = 0.65, size = 2)
  )
}

paper_compact_strip_theme <- function() {
  theme(
    axis.title = element_text(face = "bold", size = 8.2),
    axis.text = element_text(size = 6.8, colour = "black"),
    strip.text = element_text(
      face = "bold",
      size = 7.2,
      margin = margin(b = 0.5)
    ),
    legend.text = element_text(size = 7.4),
    legend.spacing.x = unit(0.04, "cm"),
    panel.spacing.x = unit(0.08, "cm"),
    plot.margin = margin(1, 4, 0, 2)
  )
}

GeomPaperPoint <- ggproto(
  "GeomPaperPoint",
  GeomPoint,
  draw_panel = function(data, panel_params, coord, na.rm = FALSE) {
    coords <- coord$transform(data, panel_params)
    grobs <- lapply(seq_len(nrow(coords)), function(i) {
      if (coords$shape[i] %in% c(26, 27, 28)) {
        polygon_spec <- switch(
          as.character(coords$shape[i]),
          "26" = list(vertices = 6, rotation = pi / 6),
          "27" = list(vertices = 5, rotation = pi / 2),
          "28" = list(vertices = 3, rotation = -pi / 2)
        )
        angles <- seq(
          0,
          2 * pi,
          length.out = polygon_spec$vertices + 1
        ) + polygon_spec$rotation
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
    if (isTRUE(data$shape %in% c(26, 27, 28))) {
      polygon_spec <- switch(
        as.character(data$shape),
        "26" = list(vertices = 6, rotation = pi / 6),
        "27" = list(vertices = 5, rotation = pi / 2),
        "28" = list(vertices = 3, rotation = -pi / 2)
      )
      angles <- seq(
        0,
        2 * pi,
        length.out = polygon_spec$vertices + 1
      ) + polygon_spec$rotation
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
