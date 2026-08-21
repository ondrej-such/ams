# Multiplicative miscalibration of logistic models

This repository contains the Quarto manuscript `index.qmd` and the R scripts
that generate its figures and simulation data.

## Prerequisites

- **R** (4.x), with the following packages:

  ```r
  install.packages(c(
    "glmnet", "logistf", "arm", "MASS", "nnet",
    "tikzDevice", "reticulate"
  ))
  ```

- **Quarto CLI** (<https://quarto.org/docs/get-started/>).
- **A LaTeX distribution**, needed for the PDF output and for the `tikz`
  figures rendered via `tikzDevice`. If you don't already have one:

  ```r
  install.packages("tinytex")
  tinytex::install_tinytex()
  ```

- **Python is only needed if you are regenerating the speaker-embedding
  CSVs from scratch** (the `data_preparation.R` section near the end of
  `index.qmd`). As long as `data/embeddings_*.csv` are already present
  (they are, in this repo), `data_preparation.R` reads them directly and
  skips the audio pipeline, so Python/`reticulate`/torch are not required
  for a normal build. They're only needed if you delete `data/*.csv` and
  want to rebuild them from `recordings/` (ECAPA-TDNN via `speechbrain`;
  needs `torch`, `torchaudio`, `speechbrain`, `soundfile`, `truststore`,
  plus an ~80 MB pretrained model downloaded on first run).

## Building the PDF

From this directory:

```sh
make pdf
```

This will:

1. Run `simulate_calibration_ratios.R` to (re)generate
   `calibration_ratios.csv` if it's missing.
2. Run the three-class coupling/partial-coupling scripts and the
   Cohen's-*d* miscalibration scan (`firth_dscan.R`) to (re)generate their
   PNGs if missing.
3. Clear Quarto's `_freeze` cache (needed because `_quarto.yml` sets
   `execute: freeze: true`, which otherwise reuses stale cached chunk
   output) and render `index.qmd` to PDF via
   `quarto render index.qmd --to pdf`.

The output PDF is written under `_manuscript/`.

### Other Makefile targets

- `make csv` — just (re)generate `calibration_ratios.csv`.
- `make figures` — just (re)generate the three-class Applications figures
  and the Cohen's-*d* scan, without rendering the PDF.
- `make clean` — remove all generated CSVs, PNGs, stamp files, and the
  Quarto freeze/manuscript caches.

### Rebuilding a single figure

Each generating script can also be run directly, e.g.:

```sh
Rscript firth_dscan.R
Rscript three_class_calibration.R
```

Scripts that produce multiple output files at once are gated behind a
`.stamp` file in the Makefile (e.g. `.coupling.stamp`) so `make` runs them
once rather than once per stale output.

## Notes

- If you change a `.R` script that produces an input to `index.qmd` (e.g.
  `simulate_calibration_ratios.R`), delete the corresponding output
  (`calibration_ratios.csv`, or the relevant `.stamp` file) before
  `make pdf`, since Make otherwise treats existing outputs as up to date.
- `_quarto.yml` also defines `html`, `docx`, and `jats` formats
  (`quarto render index.qmd --to html`, etc.), though the Makefile only
  drives the PDF target.
