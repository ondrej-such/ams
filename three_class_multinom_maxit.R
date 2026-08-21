# Does multinomial logistic regression's calibration depend on maxit?
#
# Companion to three_class_calibration.R, isolating ONE model (multinomial
# softmax regression) and varying ONE knob (the iteration cap), across both
# geometries.
#
# Motivation. nnet::multinom() applies no penalty by default: it forwards to
# nnet.default(), whose signature is decay = 0, maxit = 100. Under linearly
# separable data the MLE does not exist, the coefficients diverge, and the only
# thing halting them is the iteration cap (or the reltol = 1e-8 relative
# tolerance). So at wide separation the fitted probabilities -- and therefore
# the apparent calibration -- are a property of the OPTIMIZER SETTINGS, not of
# multinomial logistic regression. This script tests that directly: if the
# histograms move as maxit goes 100 -> 500 -> 1000, the "calibration" being
# measured is an artifact rather than a property of the model.
#
# The prediction is that they move at 9-12-15 (separable, MLE nonexistent) and
# stay put at 3-4-5 (classes overlap, MLE exists, so the optimizer converges on
# its own well before any of these caps bind).
#
# Within a replication the SAME sampled dataset is used for every maxit value,
# so differences between rows are attributable to the cap alone and not to
# resampling.

suppressPackageStartupMessages(library(nnet))

n_reps <- 30      # Monte Carlo replications
n      <- 150     # points per class
K      <- 3
ln10   <- log(10)

maxit_grid <- c(100, 1000)   # 100 = nnet's documented default

geometries <- list(
  `9-12-15` = list(mus = list(c(0, 0), c(9, 0), c(0, 12)), tag = "sep9"),
  `3-4-5`   = list(mus = list(c(0, 0), c(3, 0), c(0, 4)),  tag = "sep3")
)

logsumexp <- function(v) {
  m <- max(v)
  m + log(sum(exp(v - m)))
}
log_softmax_rows <- function(M) t(apply(M, 1, function(v) v - logsumexp(v)))

## ---- One replication: one dataset, fitted at every maxit ------------------
fit_one_rep <- function(rep_id, mus) {
  Xs <- lapply(mus, function(m) cbind(rnorm(n, m[1], 1), rnorm(n, m[2], 1)))
  X  <- do.call(rbind, Xs)
  y  <- factor(rep(1:K, each = n))
  dat <- data.frame(x = X[, 1], y2 = X[, 2], class = y)
  own <- as.integer(y)

  ## Ground truth: exact Bayes posterior from the TRUE means (equal priors,
  ## identity covariance -> log density is -0.5 * squared distance).
  logf <- sapply(mus, function(m)
    -0.5 * rowSums((X - matrix(m, nrow(X), 2, byrow = TRUE))^2))
  log_p_true <- log_softmax_rows(logf)

  do.call(rbind, lapply(maxit_grid, function(mi) {
    fit <- suppressWarnings(multinom(class ~ x + y2, data = dat,
                                     trace = FALSE, maxit = mi))
    ## Softmax the raw linear predictors ourselves rather than taking
    ## fitted()/predict(type="probs"): those are linear-scale and round to
    ## exactly 1 or 0 at this separation, which would erase the very
    ## differences we are trying to measure.
    B   <- rbind(0, coef(fit))          # reference class 1 -> zero row
    eta <- cbind(1, X) %*% t(B)
    lp  <- log_softmax_rows(eta)

    data.frame(
      rep         = rep_id,
      maxit       = mi,
      ## vmmin's ifail: 0 = converged on abstol/reltol, 1 = hit the cap.
      hit_cap     = as.integer(isTRUE(fit$convergence == 1)),
      max_abs_coef = max(abs(coef(fit))),
      deviance    = fit$deviance,
      point_id    = seq_along(own),
      class       = own,
      log10_ratio = (lp[cbind(seq_along(own), own)] -
                     log_p_true[cbind(seq_along(own), own)]) / ln10
    )
  }))
}

## ---- Diagnostics page: why raising maxit changes nothing ------------------
## The histograms alone only show that the two caps agree; they do not say why.
## These four panels distinguish the two possible explanations:
##   (a) the optimizer is genuinely converging before either cap binds, versus
##   (b) the cap binds but the extra iterations move nothing measurable.
## Under separation the deviance keeps decreasing forever, so if the fits are
## stopping early it must be on vmmin's reltol = 1e-8 relative tolerance -- the
## relative improvement per step collapses long before the coefficients do.
plot_diagnostics <- function(res, geom_name, tag) {
  lo <- min(maxit_grid); hi <- max(maxit_grid)

  ## Per-replication summaries (one row per rep x maxit).
  d <- res[!duplicated(res[, c("rep", "maxit")]), ]
  d_lo <- d[d$maxit == lo, ]; d_hi <- d[d$maxit == hi, ]
  d_lo <- d_lo[order(d_lo$rep), ]; d_hi <- d_hi[order(d_hi$rep), ]

  ## Per-point log-ratios, matched on (rep, point_id) so the difference is
  ## strictly like-for-like: same replication, same sampled point.
  m <- merge(res[res$maxit == lo, c("rep", "point_id", "log10_ratio")],
             res[res$maxit == hi, c("rep", "point_id", "log10_ratio")],
             by = c("rep", "point_id"), suffixes = c(".lo", ".hi"))
  dif <- m$log10_ratio.hi - m$log10_ratio.lo

  out_png <- file.path(getwd(), sprintf("three_class_multinom_maxit_%s_diagnostics.png", tag))
  png(out_png, width = 10, height = 8.5, units = "in", res = 150)
  par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3.5, 1), oma = c(0, 0, 3, 0))

  ## (a) Did the optimizer stop at the cap, or on its own tolerance?
  frac <- sapply(maxit_grid, function(mi) mean(d$hit_cap[d$maxit == mi]))
  bp <- barplot(frac, names.arg = paste("maxit =", maxit_grid), ylim = c(0, 1),
                col = "grey80", border = "white", ylab = "fraction of replications",
                main = "Stopped at the iteration cap?")
  text(bp, frac, sprintf("%.0f%%", 100 * frac), pos = 3, cex = 0.9)
  mtext("0% => vmmin converged on reltol/abstol, so the cap never bound",
        side = 1, line = 3, cex = 0.7)

  ## (b) Coefficient magnitude: under separation this grows without bound, so
  ## equality here means the extra iterations were never taken.
  rng <- range(c(d_lo$max_abs_coef, d_hi$max_abs_coef))
  plot(d_lo$max_abs_coef, d_hi$max_abs_coef, log = "xy", pch = 16, cex = 0.8,
       col = "#377eb8", xlim = rng, ylim = rng,
       xlab = sprintf("max |coef|, maxit = %d", lo),
       ylab = sprintf("max |coef|, maxit = %d", hi),
       main = "Coefficient magnitude")
  abline(0, 1, col = "red", lty = 2, lwd = 2)

  ## (c) Deviance: likewise strictly decreasing in the iteration count.
  rng <- range(c(d_lo$deviance, d_hi$deviance))
  plot(d_lo$deviance, d_hi$deviance, pch = 16, cex = 0.8, col = "#4daf4a",
       xlim = rng, ylim = rng,
       xlab = sprintf("deviance, maxit = %d", lo),
       ylab = sprintf("deviance, maxit = %d", hi),
       main = "Residual deviance")
  abline(0, 1, col = "red", lty = 2, lwd = 2)

  ## (d) The actual quantity of interest, differenced point by point.
  if (all(dif == 0)) {
    plot.new()
    title(main = "Per-point change in log10(pred/true)")
    text(0.5, 0.55, "Identical for every point", cex = 1.2)
    text(0.5, 0.4, sprintf("(all %d differences exactly 0)", length(dif)), cex = 0.9)
  } else {
    hist(dif, breaks = 40, col = "grey85", border = "white", freq = FALSE,
         xlab = sprintf("log10 ratio at maxit=%d  minus  maxit=%d", hi, lo),
         main = "Per-point change in log10(pred/true)")
    abline(v = 0, col = "red", lwd = 2, lty = 2)
    mtext(sprintf("max |difference| = %.3g", max(abs(dif))), side = 3,
          line = 0.2, cex = 0.75)
  }

  mtext(sprintf("Multinomial LR, %s geometry: why maxit %d -> %d changes nothing",
                geom_name, lo, hi), outer = TRUE, cex = 1.05, font = 2)
  invisible(dev.off())
  cat("Wrote", out_png, "\n")

  cat(sprintf("\nMax |per-point difference| in log10 ratio between maxit %d and %d: %.3g\n",
              lo, hi, max(abs(dif))))
  cat(sprintf("Deviance identical in %d of %d replications\n",
              sum(d_lo$deviance == d_hi$deviance), nrow(d_lo)))
}

## ---- Run one geometry end to end ------------------------------------------
run_geometry <- function(geom_name, geom) {
  cat(sprintf("\n================ geometry %s ================\n", geom_name))
  set.seed(1)
  res <- do.call(rbind, lapply(seq_len(n_reps), fit_one_rep, mus = geom$mus))

  ## Per-replication diagnostics (one row per rep x maxit, not per point).
  d <- res[!duplicated(res[, c("rep", "maxit")]), ]
  cat("\nFraction of replications stopping at the iteration cap:\n")
  print(round(tapply(d$hit_cap, d$maxit, mean), 3))
  cat("\nMedian largest |coefficient| (diverges without bound under separation):\n")
  print(signif(tapply(d$max_abs_coef, d$maxit, median), 4))

  ok <- is.finite(res$log10_ratio)
  cat("\nMean log10(p_hat / p_true), rows = maxit, cols = class:\n")
  print(signif(tapply(res$log10_ratio[ok],
                      list(res$maxit[ok], res$class[ok]), mean), 3))
  cat("\nSD of log10(p_hat / p_true):\n")
  print(signif(tapply(res$log10_ratio[ok],
                      list(res$maxit[ok], res$class[ok]), sd), 3))
  cat("\n(If these tables change down the maxit rows, the calibration being\n")
  cat(" measured is a property of the optimizer cap, not of the model.)\n")

  out_csv <- file.path(getwd(), sprintf("three_class_multinom_maxit_%s.csv", geom$tag))
  write.csv(res, out_csv, row.names = FALSE)
  cat("Wrote", out_csv, "\n")

  ## ---- Plot: maxit rows x class columns -----------------------------------
  out_png <- file.path(getwd(), sprintf("three_class_multinom_maxit_%s.png", geom$tag))
  png(out_png, width = 10, height = 8.5, units = "in", res = 150)
  par(mfrow = c(length(maxit_grid), K), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))
  for (mi in maxit_grid) {
    for (k in 1:K) {
      v <- res$log10_ratio[res$maxit == mi & res$class == k]
      v <- v[is.finite(v)]
      h <- hist(v, breaks = 40, plot = FALSE)
      ymax <- max(h$density) * 1.15
      hist(v, breaks = h$breaks, freq = FALSE, col = "grey85", border = "white",
           xlab = "log10(predicted / true)", ylim = c(0, ymax), cex.axis = 0.8,
           main = sprintf("maxit = %d\nclass %d", mi, k))
      abline(v = 0, col = "red", lwd = 2, lty = 2)
      mval <- mean(v)
      arrows(x0 = mval, y0 = ymax * 0.65, x1 = mval, y1 = ymax * 0.45,
             col = "blue", lwd = 2, length = 0.08)
      lab <- sprintf("mean=%.1e", mval)
      usr <- par("usr")
      half <- strwidth(lab, cex = 0.85) / 2
      x_lab <- min(max(mval, usr[1] + half * 1.05), usr[2] - half * 1.05)
      text(x_lab, ymax * 0.65, lab, col = "blue", pos = 3, cex = 0.85)
    }
  }
  mtext(sprintf("Multinomial LR (decay = 0, unpenalised): %s geometry, M = %d replications",
                geom_name, n_reps), outer = TRUE, cex = 1.0, font = 2)
  invisible(dev.off())
  cat("Wrote", out_png, "\n")

  ## Second page, for the separable geometry only: the histograms above show
  ## the two caps agree, but not why.
  if (geom$tag == "sep9") plot_diagnostics(res, geom_name, geom$tag)

  invisible(res)
}

for (gname in names(geometries)) run_geometry(gname, geometries[[gname]])
