args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: plot_components.R /study/output /new/plot/output")
}
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
source(file.path(dirname(normalizePath(script)), "../../paper/figures/paper_plot_style.R"))
library(tikzDevice)

input <- normalizePath(args[1], mustWork = TRUE)
output <- args[2]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
summary <- read.csv(file.path(input, "summary.csv"), na.strings = c("", "NA"))
curves <- read.csv(file.path(input, "curves.csv"), na.strings = c("", "NA"))
contrasts <- read.csv(file.path(input, "contrasts.csv"), na.strings = c("", "NA"))
profiles <- unique(summary$profile)
if (length(profiles) != 1 || !profiles %in% c("full", "pilot")) {
  stop("Expected one explicit full or pilot profile")
}
note <- if (profiles == "pilot") {
  "PREFIX PILOT ONLY - not a YouTube1M result"
} else {
  "YouTube1M, raw IP, original mixed-AND queries, recall@10"
}
missing_note <- "Missing QPS95 points: incomplete batches or no supported target bracket."

variants <- c("baseline", "c1", "c1_c2", "full", "without_c1")
labels <- c("Baseline", "+C1", "+C1+C2", "EMA", "EMA without C1")
colours <- c("#777777", "#104680", "#317D5A", "#B72230", "#6A3D9A")
strict_comparison <- "strict" %in% summary$variant
if (strict_comparison) {
  variants <- c("strict", "c1_c2", "full")
  labels <- c("Strict in-filtering", "Marker only", "EMA")
  colours <- c("#777777", "#317D5A", "#B72230")
  note <- paste(note, "- same CHT-on graph")
}
names(colours) <- labels
summary$variant <- factor(summary$variant, levels = variants, labels = labels)
curves$variant <- factor(curves$variant, levels = variants, labels = labels)
summary$selectivity <- summary$nominal_selectivity * 100
contrasts$selectivity <- contrasts$nominal_selectivity * 100
ticks <- sort(unique(summary$selectivity))

style <- function() {
  list(
    paper_theme(9),
    scale_colour_manual(values = colours, drop = FALSE),
    guides(colour = paper_guide()),
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )
}

save_plot <- function(plot, name, width = 7, height = 3.3) {
  ggsave(file.path(output, paste0(name, ".pdf")), plot,
         width = width, height = height, device = cairo_pdf)
  tikz(file.path(output, paste0(name, ".tex")), width = width, height = height,
       standAlone = FALSE, sanitize = TRUE)
  on.exit(dev.off())
  print(plot)
}

for (metric in c("qps95", "observed_qps")) {
  summary$value <- as.numeric(summary[[metric]])
  label <- if (metric == "qps95") {
    "QPS at 95% recall (supported brackets only)"
  } else {
    "QPS at first sampled ef reaching at least 95% recall"
  }
  plot <- ggplot(summary, aes(selectivity, value, colour = variant)) +
    geom_line(linewidth = 0.55, na.rm = TRUE) + geom_point(size = 1.8, na.rm = TRUE) +
    scale_x_log10(breaks = ticks) + scale_y_log10() +
    labs(x = "Nominal selectivity (%)", y = label,
         caption = if (metric == "qps95") paste(note, missing_note, sep = "\n") else note) +
    style()
  save_plot(plot, metric)
}

component_names <- c(
  "C1_without_Marker" = "C1 without Marker",
  "C2_without_recovery" = "C2 without recovery",
  "C3_with_Marker" = "C3 with Marker",
  "C1_in_full_method" = "C1 in full method",
  "full_vs_baseline" = "Full method / baseline"
)
if (strict_comparison) {
  component_names <- c(
    "Marker_vs_strict" = "Marker / strict",
    "full_vs_strict" = "EMA / strict",
    "recovery_vs_Marker" = "EMA / Marker"
  )
}
contrasts$component <- factor(contrasts$component, levels = names(component_names),
                              labels = unname(component_names))
for (name in c("paired_qps95_speedup", "paired_qps95_speedup_min", "paired_qps95_speedup_max")) {
  contrasts[[name]] <- as.numeric(contrasts[[name]])
}
plot <- ggplot(contrasts, aes(selectivity, paired_qps95_speedup)) +
  geom_hline(yintercept = 1, colour = "#777777", linetype = "dashed") +
  geom_linerange(aes(ymin = paired_qps95_speedup_min, ymax = paired_qps95_speedup_max),
                 linewidth = 0.3, colour = "#777777", na.rm = TRUE) +
  geom_line(linewidth = 0.5, na.rm = TRUE) +
  geom_point(size = 1.5, na.rm = TRUE) +
  facet_wrap(~component, ncol = 3) +
  scale_x_log10(breaks = c(1, 5, 20, 100)) + scale_y_log10() +
  labs(x = "Nominal selectivity (%)", y = "Paired QPS95 ratio (>1 is a gain)",
       caption = paste(note, missing_note,
                       "Ranges are observed paired-round min/max, not confidence intervals.",
                       sep = "\n")) +
  paper_theme(8.5)
save_plot(plot, "component_speedups", height = 4.5)

curves$cell <- factor(curves$cell, levels = paste0("T", ticks),
                      labels = paste0(ticks, "%"))
recall_min <- max(0, floor(min(curves$recall, 0.95) * 20) / 20)
recall_ticks <- sort(unique(c(seq(ceiling(recall_min * 10) / 10, 1, 0.1), 0.95)))
if (strict_comparison && 1 - recall_min > 0.35) {
  recall_ticks <- sort(unique(c(seq(ceiling(recall_min * 5) / 5, 0.8, 0.2), 0.95)))
}
plot <- ggplot(curves, aes(recall, qps, colour = variant)) +
  geom_vline(xintercept = 0.95, colour = "#999999", linetype = "dashed") +
  geom_path(linewidth = 0.5) + geom_point(size = 1.2) +
  facet_wrap(~cell, ncol = 4, scales = "free_y", drop = FALSE) +
  scale_x_continuous(limits = c(recall_min, 1), breaks = recall_ticks) +
  scale_y_log10() +
  labs(x = "Recall@10", y = "QPS",
       caption = paste(note, "Missing curves: no complete returned batch at any sampled ef.",
                       sep = "\n")) + style()
save_plot(plot, "recall_qps", height = 6)
