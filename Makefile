QMD := index.qmd
CSV := calibration_ratios.csv

## PNGs produced by the two three-class pairwise-coupling scripts (see the
## Applications section). Each script writes several files in one run, so
## we route them through a stamp file rather than giving each PNG its own
## recipe -- otherwise Make would re-run the (slow-ish) script once per
## stale output file instead of once total. This also keeps things
## portable to older Make versions that lack grouped-target ("&:") rules.
COUPLING_PNGS := three_class_firth_coupling_rgb.png \
                 three_class_firth_coupling_argmax.png \
                 three_class_firth_coupling_probs.png \
                 three_class_ht_coupling_rgb.png \
                 three_class_ht_coupling_argmax.png \
                 three_class_lda_rgb.png \
                 three_class_lda_argmax.png \
                 three_class_lda_wlw_coupling_rgb.png \
                 three_class_lda_wlw_coupling_argmax.png \
                 three_class_lda_ht_coupling_rgb.png \
                 three_class_lda_ht_coupling_argmax.png

PARTIAL_PNGS := three_class_firth_partial_wlw_drop12_argmax.png \
                three_class_firth_partial_wlw_drop13_argmax.png \
                three_class_firth_partial_wlw_drop23_argmax.png \
                three_class_firth_full_wlw_argmax_reference.png

## PNGs from the LDA one-vs-one analogue of the partial-coupling experiment.
LDA_PARTIAL_PNGS := three_class_lda_partial_wlw_drop12_argmax.png \
                    three_class_lda_partial_wlw_drop13_argmax.png \
                    three_class_lda_partial_wlw_drop23_argmax.png \
                    three_class_lda_full_wlw_argmax_reference.png

## Decision-boundary overlay: Firth + Hastie-Tibshirani vs. Wu-Lin-Weng.
HT_BOUNDARY_PNG := three_class_firth_ht_boundaries.png

## Multiclass calibration histograms, at both separations (9-12-15 and 3-4-5).
CALIB_OUT := three_class_calibration_hists_sep9.png \
             three_class_calibration_hists_sep3.png \
             three_class_calibration_sep9.csv \
             three_class_calibration_sep3.csv

## Firth miscalibration vs Cohen's d (evaluated at the mode of class A).
DSCAN_OUT := firth_dscan.png firth_dscan.csv

## Multinomial-only maxit sensitivity check, both separations.
MAXIT_OUT := three_class_multinom_maxit_sep9.png \
             three_class_multinom_maxit_sep3.png \
             three_class_multinom_maxit_sep9_diagnostics.png \
             three_class_multinom_maxit_sep9.csv \
             three_class_multinom_maxit_sep3.csv

.PHONY: all pdf clean csv figures

all: pdf

## Clear Quarto's freeze cache, then render the PDF. Freeze cache removal is
## necessary because _quarto.yml has `execute: freeze: true`, which makes
## Quarto reuse old cached chunk output instead of re-running R code unless
## the cache is gone. Depends on the CSV and the three-class PNGs since the
## paper reads/embeds them directly.
pdf: $(CSV) $(COUPLING_PNGS) $(PARTIAL_PNGS) $(LDA_PARTIAL_PNGS) $(DSCAN_OUT)
	rm -rf _freeze .quarto/_freeze
	quarto render $(QMD) --to pdf

## Run the calibration-ratio Monte Carlo simulation (N=50/class, means 6
## apart, unit variance), fitting L1/L2/Firth/Platt/Platt-2N/LDA and
## recording true-vs-predicted probability ratios, into calibration_ratios.csv.
csv: $(CSV)

$(CSV): simulate_calibration_ratios.R
	Rscript simulate_calibration_ratios.R

## Regenerate the three-class Applications figures.
figures: .coupling.stamp .partial.stamp .lda_partial.stamp $(HT_BOUNDARY_PNG) \
         .calibration.stamp .maxit.stamp .dscan.stamp

$(COUPLING_PNGS): .coupling.stamp
.coupling.stamp: three_class_firth_coupling.R
	Rscript three_class_firth_coupling.R
	touch .coupling.stamp

$(PARTIAL_PNGS): .partial.stamp
.partial.stamp: three_class_firth_partial_wlw.R
	Rscript three_class_firth_partial_wlw.R
	touch .partial.stamp

$(LDA_PARTIAL_PNGS): .lda_partial.stamp
.lda_partial.stamp: three_class_lda_partial_wlw.R
	Rscript three_class_lda_partial_wlw.R
	touch .lda_partial.stamp

$(HT_BOUNDARY_PNG): three_class_firth_ht_boundaries.R
	Rscript three_class_firth_ht_boundaries.R

$(CALIB_OUT): .calibration.stamp
.calibration.stamp: three_class_calibration.R
	Rscript three_class_calibration.R
	touch .calibration.stamp

$(MAXIT_OUT): .maxit.stamp
.maxit.stamp: three_class_multinom_maxit.R
	Rscript three_class_multinom_maxit.R
	touch .maxit.stamp

$(DSCAN_OUT): .dscan.stamp
.dscan.stamp: firth_dscan.R
	Rscript firth_dscan.R
	touch .dscan.stamp

## Remove all cached/generated build artifacts.
clean:
	rm -rf _freeze .quarto/_freeze _manuscript $(CSV) \
	       $(COUPLING_PNGS) $(PARTIAL_PNGS) $(LDA_PARTIAL_PNGS) \
	       $(HT_BOUNDARY_PNG) $(CALIB_OUT) $(MAXIT_OUT) $(DSCAN_OUT) \
	       .coupling.stamp .partial.stamp .lda_partial.stamp \
	       .calibration.stamp .maxit.stamp .dscan.stamp
