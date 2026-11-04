# Publication sources

This directory contains active appendix dependencies and the bundled
main-figure generators, including the plot formerly numbered Figure 7 (the
current manuscript calls the three-operation dynamic result Figure 9).
It does not import the unused chapter copies
or old attribute-only, tombstone/patch, or four-operation dynamic plots from the
authoring workspace.

## Build the appendix

The appendix's native implementation dependencies are published in `hnswlib/`
and `exp_benchmark/` before updating the repository's top-level PDF.
With a LaTeX installation providing the ACM class dependencies and `latexmk`,
run from the repository root:

```bash
cd paper/appendix
latexmk -pdf -interaction=nonstopmode -halt-on-error appendix.tex
cp appendix.pdf ../../EMA_appendix.pdf
```

`appendix/APP/app.tex` is the active content. The bundled class, bibliography
style, bibliography, six active figure PDFs, generated summary files, and
the accompanying CSV/provenance snapshots in `appendix/CH5/figures/data`
are included; no document inputs come from the external manuscript workspace.

The current report includes two-layer construction and local deletion analysis,
source-specific false-positive bounds, the dynamic evaluation protocol, speedup
accounting, the complete construction policy, and expanded ablation and parameter
studies. The current main paper's three-operation dynamic result is Figure 9;
its plotting input is in `figures/data/ema_dynamic_v11.csv`. The frozen V11
implementation is not bundled. Historical paths in the plotting provenance
identify the original inputs, not files available in this checkout.
Paths under `paper/CH5/figures`
in the report refer to the manuscript authoring layout, not this repository's
`paper/figures` layout.

This is a separately maintained technical appendix, not a claim that a
particular conference permits an appendix as part of its submission.

## Re-render current main figures

Rendering uses R and TikZ, not benchmark execution. Install the R dependencies
when setting up a rendering environment:

```r
install.packages(c(
  "ggplot2", "ggthemes", "ggh4x", "patchwork", "tikzDevice", "readxl",
  "scales", "reshape2", "plyr", "latex2exp", "tidyr"
))
```

With LaTeX/TikZ also installed, run:

```bash
cd paper/figures
Rscript range_label_combined_plot.R
Rscript single_attribute_combined_plot.R
Rscript ema_dynamic_v11_plot.R --timing=total
```

| Figure | Output |
|---|---|
| 5 | `range_label_combined_qps_tikz.tex` |
| 6 | `single_attribute_combined_qps_tikz.tex` |
| Earlier Figure 7 (current manuscript Figure 9) | `ema_dynamic_compact_tikz.tex` (also standalone QPS/maintenance TikZ views) |

The static R files contain the recorded plotting values. They do not run the
baseline systems or prove that the historical inputs/binaries are available.
Appendix figure PDFs are supplied as compilation inputs; this directory does
not provide a complete generator/raw-data chain for every appendix figure.
See the [experiment coverage map](../exp_benchmark/README.md#paper-to-code-map).

The bundled dynamic plot reads `figures/data/ema_dynamic_v11.csv`: exactly three operations,
six stages each, QPS at 95% recall, and original mutation-call wall times.
`point_update` means complete vector-and-attribute replacement via synchronous
deletion followed by insertion at fresh labels. It is not attribute-only update.
`--timing=total` is the published plot. The optional scrub view requires actual
component measurements and intentionally fails when those values are absent.
Do not fabricate them or reinterpret the historical mutation times.

The already generated TikZ files are included for rendering without R.
They are figure bodies intended for `\input{...}` inside a document loading
`tikz`; they are not standalone LaTeX documents.
