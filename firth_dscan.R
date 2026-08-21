# Firth-regression miscalibration as a function of class separation (Cohen's d).
#
# For each d in [1, 8]: draw n = 50 points per class from N(0,1) and N(d,1)
# (unit variance, so the distance between means IS Cohen's d), fit Firth
# logistic regression, and evaluate the predicted probability of the LESS
# LIKELY class at the mode of class A (x = 0). At that point the less likely
# class is B, with true probability p_true = sigma(-d^2/2), tiny for large d.
# The quantity plotted is
#
#   log10( p_hat_Firth(B | x = 0) / p_true(B | x = 0) ),
#
# averaged over n_reps Monte Carlo replications (band = +/- 1 sd). This is
# the "multiplicative miscalibration at the class mode" as a function of
# effect size, addressing the question of whether the phenomenon requires
# extreme separation (d = 6 in the main simulation) or is already present
# at commonly encountered effect sizes.
#
# Everything stays on the log-odds scale via plogis(eta, log.p = TRUE), as in
# simulate_calibration_ratios.R, so no clamping is needed anywhere.

set.seed(1)

n_reps <- 100
n      <- 50
d_grid <- seq(1, 8, by = 0.25)

## Firth logistic regression via modified-score IRLS (same algorithm as the
## multi-dimensional version in the three_class_* scripts, here with a
## 1-D design). Self-contained: no logistf dependency.
firth_logistic_1d <- function(x, y, tol = 1e-8, max_iter = 100) {
  X <- cbind(1, x)
  beta <- c(0, 0)
  for (i in 1:max_iter) {
    eta  <- as.vector(X %*% beta)
    p    <- plogis(eta)
    W    <- p * (1 - p)
    XtWX <- t(X) %*% (X * W)
    XtWX_inv <- solve(XtWX)
    h <- W * rowSums((X %*% XtWX_inv) * X)
    U_star <- t(X) %*% (y - p + h * (0.5 - p))
    step <- as.vector(XtWX_inv %*% U_star)
    beta <- beta + step
    if (max(abs(step)) < tol) break
  }
  beta
}

ln10 <- log(10)

res <- do.call(rbind, lapply(d_grid, function(d) {
  per_rep <- vapply(seq_len(n_reps), function(rep_id) {
    x <- c(rnorm(n, 0, 1), rnorm(n, d, 1))
    y <- rep(c(0, 1), each = n)
    beta <- firth_logistic_1d(x, y)
    ## Predicted and true log P(B | x = 0); x = 0 is the mode of class A,
    ## where B is the less likely class. eta_true(0) = -d^2/2.
    log_p_hat  <- plogis(beta[1], log.p = TRUE)
    log_p_true <- plogis(-d^2 / 2, log.p = TRUE)
    ## beta[2] is the fitted log odds ratio per 1 SD of x (the true value
    ## under the generative model is d, i.e. true OR = exp(d)).
    c(ratio = (log_p_hat - log_p_true) / ln10, log_or = beta[2])
  }, c(ratio = 0, log_or = 0))
  data.frame(d = d,
             mean_log10_ratio = mean(per_rep["ratio", ]),
             sd_log10_ratio   = sd(per_rep["ratio", ]),
             mean_log10_or    = mean(per_rep["log_or", ]) / ln10,
             true_log10_or    = d / ln10)
}))

out_dir <- path.expand("~/ams")
write.csv(res, file.path(out_dir, "firth_dscan.csv"), row.names = FALSE)

png(file.path(out_dir, "firth_dscan.png"),
    width = 7.5, height = 5.5, units = "in", res = 150)
par(mar = c(4.5, 4.5, 4, 4.5))
ylim <- range(res$mean_log10_ratio - res$sd_log10_ratio,
              res$mean_log10_ratio + res$sd_log10_ratio, 0,
              res$true_log10_or)
plot(res$d, res$mean_log10_ratio, type = "n", ylim = ylim,
     xlab = expression("Cohen's " * italic(d) * " between the two classes"),
     ylab = "log10(predicted prob / true prob) at mode of class A")
## Draw the miscalibration curve/band only up to d = 7.2 so it does not run
## into the odds-ratio legend in the upper right corner (the CSV still
## contains the full grid).
shown <- res$d <= 7.2
polygon(c(res$d[shown], rev(res$d[shown])),
        c((res$mean_log10_ratio - res$sd_log10_ratio)[shown],
          rev((res$mean_log10_ratio + res$sd_log10_ratio)[shown])),
        col = adjustcolor("#7570b3", alpha.f = 0.25), border = NA)
lines(res$d[shown], res$mean_log10_ratio[shown], lwd = 2, col = "#7570b3")
abline(h = 0, col = "red", lwd = 2, lty = 2)

## Odds ratios, drawn on the same log10 scale (an OR is just exp of a log
## odds, so log10(OR) lives naturally on this axis): the true OR per 1 SD
## of x, exp(d), and the mean Firth-fitted OR, exp(beta1). The gap that
## opens between them past the separability threshold is the same
## information loss seen in the calibration curve, viewed on the OR scale.
lines(res$d, res$true_log10_or, lwd = 2, lty = 3, col = "black")
lines(res$d, res$mean_log10_or, lwd = 2, col = "#1b9e77")
axis(4, at = log10(c(3, 30, 300, 3000)),
     labels = c("3", "30", "300", "3000"), las = 1)
mtext(expression("odds ratio per 1 SD of " * italic(x) * " (log scale)"),
      side = 4, line = 3)

## Top axis: the true OR corresponding to each Cohen's d, for readers who
## think in odds ratios rather than standardized mean differences.
d_ticks <- 1:8
axis(3, at = d_ticks, labels = sprintf("%.0f", exp(d_ticks)))
mtext(expression("true OR corresponding to " * italic(d)), side = 3, line = 2.2)

## Split legend: calibration-related entries on the left, OR-related entries
## on the right (both anchored at the top of the plot region).
legend("topleft", bty = "n", cex = 0.9,
       title = "Calibration (left axis)", title.adj = 0,
       legend = c("miscalibration, Firth (mean of 100 reps)",
                  expression(""%+-%"1 sd"),
                  "perfect calibration"),
       col = c("#7570b3", adjustcolor("#7570b3", alpha.f = 0.25), "red"),
       lty = c(1, NA, 2), lwd = c(2, NA, 2),
       pch = c(NA, 15, NA), pt.cex = c(NA, 2, NA))
legend("topright", bty = "n", cex = 0.9,
       title = "Odds ratio (right axis)", title.adj = 0,
       legend = c(expression("true OR = exp(" * italic(d) * ")"),
                  "Firth fitted OR"),
       col = c("black", "#1b9e77"),
       lty = c(3, 1), lwd = c(2, 2))
invisible(dev.off())
cat("Wrote", file.path(out_dir, "firth_dscan.png"), "and firth_dscan.csv\n")
