# Calibration of four MULTICLASS classifiers on the 3-class problem.
#
# Setup as in the other three_class_* scripts: bivariate Normal classes,
# identity covariance, means forming a right triangle, 150 points per class,
# repeated over n_reps Monte Carlo samples.
#
# TWO GEOMETRIES are run, because the answer depends entirely on separation:
#   * 9-12-15  -- the geometry used by the coupling figures elsewhere in this
#                 project. Nearest means are 9 sd apart, so the true posterior
#                 is 1 to within machine epsilon at essentially every sampled
#                 point. Calibration is then unmeasurable for any model whose
#                 predictions also saturate: the log-ratio is exactly 0 up to
#                 double-precision round-off, and the histograms show noise at
#                 the 1e-16 level rather than model behaviour. Kept because
#                 documenting that ceiling is itself informative.
#   * 3-4-5    -- same shape scaled down, nearest means 3 sd apart. The classes
#                 now overlap enough that true posteriors span a resolvable
#                 range, so differences between the models are real.
#
# Models compared (all producing a full 3-class probability vector):
#   1. LDA, fit natively on all three classes at once
#   2. Multinomial (softmax) logistic regression, also native multiclass
#   3. One-vs-one Firth logistic regression + Wu-Lin-Weng coupling
#   4. One-vs-one LDA + Wu-Lin-Weng coupling
#
# Ground truth is the exact Bayes-optimal posterior computed from the TRUE
# generating means (equal priors, identity covariance).
#
# For each (model, class k) pair we histogram log10(p_hat(k) / p_true(k)),
# evaluated at the sampled points that actually belong to class k. So these
# panels show how much probability mass each model places on the CORRECT class
# relative to the Bayes rule: zero means matching Bayes, negative means the
# model hedges more than Bayes does.
#
# Everything is computed on the LOG scale via logsumexp: at wide separation a
# linear-scale posterior rounds to exactly 1 or 0 in double precision, which
# would silently turn the quantity of interest into log(1) = 0 for every model
# and hide all the differences. The WLW coupling is the one component that
# cannot be moved into log space -- it is defined by an iteration on the
# probability simplex -- but its inputs can still be built without precision
# loss (see the note on filling both directions of each pairwise comparison
# below), which is what keeps its output meaningful here.

suppressPackageStartupMessages({
  library(MASS)   # lda
  library(nnet)   # multinom
})

n_reps <- 30      # Monte Carlo replications
n      <- 150     # points per class
K      <- 3

## The two geometries. Each is a right triangle with the right angle at class
## 1; the leg lengths are the nearest-neighbour separations in units of sd.
geometries <- list(
  `9-12-15` = list(mus = list(c(0, 0), c(9, 0), c(0, 12)), tag = "sep9"),
  `3-4-5`   = list(mus = list(c(0, 0), c(3, 0), c(0, 4)),  tag = "sep3")
)
ln10 <- log(10)
## Floor for genuine double-precision underflow only (plogis(-eta) underflows
## to 0 once eta exceeds about 745). Far below the 1e-10 clamp the earlier
## scripts needed, because we never form a losing probability by subtraction.
clamp_eps_floor <- 1e-300

logsumexp <- function(v) {
  m <- max(v)
  m + log(sum(exp(v - m)))
}
## Row-wise: matrix of log-scores -> matrix of log-probabilities.
log_softmax_rows <- function(M) {
  t(apply(M, 1, function(v) v - logsumexp(v)))
}

## ---- Firth logistic regression (multi-dim design matrix) ------------------
firth_logistic <- function(X, y, tol = 1e-8, max_iter = 50) {
  X <- cbind(Intercept = 1, X)
  beta <- rep(0, ncol(X))
  for (i in 1:max_iter) {
    eta  <- as.vector(X %*% beta)
    p_i  <- plogis(eta)
    W    <- p_i * (1 - p_i)
    XtWX <- t(X) %*% (X * W)
    XtWX_inv <- solve(XtWX)
    h <- W * rowSums((X %*% XtWX_inv) * X)
    U_star <- t(X) %*% (y - p_i + h * (0.5 - p_i))
    step <- as.vector(XtWX_inv %*% U_star)
    beta <- beta + step
    if (max(abs(step)) < tol) break
  }
  beta
}

## ---- Wu, Lin & Weng (2004) coupling ---------------------------------------
## eps is an ABSOLUTE tolerance on max|Qp - pQp|. WLW's published default is
## 0.005/k, which is fine when the coupled probabilities are only ever read at
## argmax resolution, but far too loose here: it leaves an error of order 1e-4
## in log10(p) whenever the pairwise log-odds are moderate (|eta| ~ 2-8), which
## is precisely the regime Firth's shrinkage puts us in, and is a few percent
## of the effect being measured. Tightened to 1e-12; this costs a handful of
## extra iterations and changes nothing once |eta| >~ 12, where the iteration
## already converges on the first pass.
wlw_couple <- function(r, max_iter = 200, eps = 1e-12) {
  k <- nrow(r)
  Q <- matrix(0, k, k)
  for (t in 1:k) {
    Q[t, t] <- sum(r[-t, t]^2)
    for (j in setdiff(1:k, t)) Q[t, j] <- -r[j, t] * r[t, j]
  }
  p <- rep(1 / k, k)
  for (iter in 1:max_iter) {
    Qp  <- as.vector(Q %*% p)
    pQp <- sum(p * Qp)
    if (max(abs(Qp - pQp)) < eps) break
    for (t in 1:k) {
      diff  <- (-Qp[t] + pQp) / Q[t, t]
      p[t]  <- p[t] + diff
      pQp   <- (pQp + diff * (diff * Q[t, t] + 2 * Qp[t])) / (1 + diff)^2
      Qp    <- (Qp + diff * Q[t, ]) / (1 + diff)
      p     <- p / (1 + diff)
    }
  }
  p / sum(p)
}

## ---- One replication -------------------------------------------------------
fit_one_rep <- function(rep_id, mus) {
  Xs <- lapply(mus, function(m) cbind(rnorm(n, m[1], 1), rnorm(n, m[2], 1)))
  X  <- do.call(rbind, Xs)
  y  <- factor(rep(1:K, each = n))
  dat <- data.frame(x = X[, 1], y2 = X[, 2], class = y)

  ## --- Ground truth: exact Bayes posterior from the TRUE means -------------
  ## Equal priors + identity covariance -> log density is -0.5 * squared dist.
  logf <- sapply(mus, function(m)
    -0.5 * rowSums((X - matrix(m, nrow(X), 2, byrow = TRUE))^2))
  log_p_true <- log_softmax_rows(logf)

  ## --- 1. LDA, native multiclass ------------------------------------------
  ## Computed from the pooled-covariance discriminant directly rather than via
  ## predict.lda()$posterior, which returns linear-scale posteriors that round
  ## to exactly 1/0 at this separation.
  mhat  <- lapply(1:K, function(k) colMeans(Xs[[k]]))
  Sigma <- Reduce(`+`, lapply(1:K, function(k) (n - 1) * cov(Xs[[k]]))) / (K * n - K)
  Sinv  <- solve(Sigma)
  delta <- sapply(1:K, function(k) {
    mk <- mhat[[k]]
    as.numeric(X %*% (Sinv %*% mk)) - 0.5 * as.numeric(t(mk) %*% Sinv %*% mk) +
      log(1 / K)
  })
  log_p_lda <- log_softmax_rows(delta)

  ## --- 2. Multinomial (softmax) logistic regression ------------------------
  ## The classes are linearly separable here, so the MLE does not exist and the
  ## coefficients diverge; multinom() stops at maxit. We keep the linear
  ## predictors and softmax them ourselves, again to stay off the linear scale.
  mn <- suppressWarnings(multinom(class ~ x + y2, data = dat,
                                  trace = FALSE, maxit = 500))
  B   <- rbind(0, coef(mn))                       # reference class 1 -> zero row
  eta <- cbind(1, X) %*% t(B)
  log_p_mn <- log_softmax_rows(eta)

  ## --- One-vs-one + WLW coupling, for two different base classifiers -------
  ## Given a function eta_pair(a, b) returning the signed log-odds of class a
  ## over class b at every point, build the pairwise matrix and couple it.
  ##
  ## Both directions of each pair are filled directly as plogis(+eta) and
  ## plogis(-eta), rather than computing one and taking 1 - it. This matters:
  ## at this separation the winning side is 1 - 1e-18, which rounds to exactly
  ## 1.0 in double precision, so the complement "1 - r" collapses to exactly 0
  ## and destroys the losing side entirely. plogis(-eta) computes that same
  ## losing probability directly and stays accurate down to ~1e-308. Since
  ## Q[t,t] = sum(r[-t,t]^2) is built precisely from those losing entries,
  ## computing them this way is what keeps Q[t,t] strictly positive, and so is
  ## what actually prevents the divide-by-zero that the clamp was there to
  ## paper over. The floor is retained only for genuine underflow (|eta|
  ## beyond about 745).
  couple_wlw <- function(eta_pair) {
    R <- array(0, dim = c(nrow(X), K, K))
    for (pr in combn(K, 2, simplify = FALSE)) {
      a <- pr[1]; b <- pr[2]
      eta_ab <- eta_pair(a, b)
      R[, a, b] <- pmax(plogis(eta_ab),  clamp_eps_floor)
      R[, b, a] <- pmax(plogis(-eta_ab), clamp_eps_floor)
    }
    log(t(vapply(seq_len(nrow(X)), function(i) wlw_couple(R[i, , ]), numeric(K))))
  }

  ## --- 3. One-vs-one Firth + WLW coupling ---------------------------------
  betas <- list()
  for (pr in combn(K, 2, simplify = FALSE)) {
    lo <- pr[1]; hi <- pr[2]
    idx <- y %in% c(lo, hi)
    betas[[paste0(lo, "_", hi)]] <-
      firth_logistic(X[idx, , drop = FALSE], as.integer(y[idx] == hi))
  }
  eta_pair_firth <- function(a, b) {
    lo <- min(a, b); hi <- max(a, b)
    eta <- as.vector(cbind(1, X) %*% betas[[paste0(lo, "_", hi)]])
    if (a == hi) eta else -eta
  }
  log_p_firth_wlw <- couple_wlw(eta_pair_firth)

  ## --- 4. One-vs-one LDA + WLW coupling ------------------------------------
  ## Each pair gets its own two-class LDA, fit only on that pair's data (so its
  ## pooled covariance and class means differ from the native multiclass fit in
  ## step 1). For two groups the LDA discriminant IS a linear log-odds, so we
  ## form it in closed form and keep it unsaturated, exactly as for Firth.
  lda_pair <- list()
  for (pr in combn(K, 2, simplify = FALSE)) {
    lo <- pr[1]; hi <- pr[2]
    Xlo <- Xs[[lo]]; Xhi <- Xs[[hi]]
    S <- ((nrow(Xlo) - 1) * cov(Xlo) + (nrow(Xhi) - 1) * cov(Xhi)) /
         (nrow(Xlo) + nrow(Xhi) - 2)
    Si <- solve(S)
    mlo <- colMeans(Xlo); mhi <- colMeans(Xhi)
    w <- as.numeric(Si %*% (mhi - mlo))
    b <- -0.5 * as.numeric(t(mhi) %*% Si %*% mhi - t(mlo) %*% Si %*% mlo) +
         log(nrow(Xhi) / nrow(Xlo))
    lda_pair[[paste0(lo, "_", hi)]] <- list(w = w, b = b)
  }
  eta_pair_lda <- function(a, b) {
    lo <- min(a, b); hi <- max(a, b)
    cf  <- lda_pair[[paste0(lo, "_", hi)]]
    eta <- as.numeric(X %*% cf$w + cf$b)
    if (a == hi) eta else -eta
  }
  log_p_lda_wlw <- couple_wlw(eta_pair_lda)

  logs <- list(LDA = log_p_lda, Multinom = log_p_mn,
               `Firth+WLW` = log_p_firth_wlw, `LDA+WLW` = log_p_lda_wlw)

  ## Keep only each point's OWN class (class k panel <- class k points).
  own <- as.integer(y)
  do.call(rbind, lapply(names(logs), function(mname) {
    lp <- logs[[mname]]
    data.frame(
      rep         = rep_id,
      model       = mname,
      class       = own,
      log_p_true  = log_p_true[cbind(seq_along(own), own)],
      log_p_hat   = lp[cbind(seq_along(own), own)],
      log10_ratio = (lp[cbind(seq_along(own), own)] -
                     log_p_true[cbind(seq_along(own), own)]) / ln10
    )
  }))
}

model_order  <- c("LDA", "Multinom", "Firth+WLW", "LDA+WLW")
model_labels <- c(LDA = "LDA (native)", Multinom = "Multinomial LR",
                  `Firth+WLW` = "Firth 1v1 + WLW", `LDA+WLW` = "LDA 1v1 + WLW")

## ---- Run one geometry end to end ------------------------------------------
run_geometry <- function(geom_name, geom) {
  cat(sprintf("\n================ geometry %s ================\n", geom_name))
  ## Reset the seed per geometry so the two runs are independent replicates of
  ## the same design rather than one continuing the other's RNG stream.
  set.seed(1)
  results <- do.call(rbind, lapply(seq_len(n_reps), fit_one_rep, mus = geom$mus))

  ## Report anything the log scale could not represent.
  n_inf <- sum(!is.finite(results$log10_ratio))
  if (n_inf > 0) {
    cat(sprintf("NOTE: %d of %d rows have non-finite log-ratio (p_hat underflowed to 0); excluded from histograms.\n",
                n_inf, nrow(results)))
  }
  ok <- is.finite(results$log10_ratio)
  cat("\nMean log10(p_hat / p_true) by model and class:\n")
  print(signif(tapply(results$log10_ratio[ok],
                      list(results$model[ok], results$class[ok]), mean), 3))
  cat("\nSD of log10(p_hat / p_true) by model and class:\n")
  print(signif(tapply(results$log10_ratio[ok],
                      list(results$model[ok], results$class[ok]), sd), 3))
  cat("\n(0 = matches Bayes; negative = less mass on the correct class than Bayes.\n")
  cat(" Values at the 1e-16 level are double-precision round-off, not signal.)\n")

  out_csv <- path.expand(sprintf("~/ams/three_class_calibration_%s.csv", geom$tag))
  write.csv(results, out_csv, row.names = FALSE)
  cat("Wrote", out_csv, "\n")

  ## ---- Plot: 4 models x 3 classes -----------------------------------------
  out_png <- path.expand(sprintf("~/ams/three_class_calibration_hists_%s.png", geom$tag))
  png(out_png, width = 10, height = 11, units = "in", res = 150)
  par(mfrow = c(4, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2.5, 0))
  for (m in model_order) {
    for (k in 1:K) {
      v <- results$log10_ratio[results$model == m & results$class == k]
      v <- v[is.finite(v)]
      h <- hist(v, breaks = 40, plot = FALSE)
      ymax <- max(h$density) * 1.15
      hist(v, breaks = h$breaks, freq = FALSE, col = "grey85", border = "white",
           xlab = "log10(predicted / true)", ylim = c(0, ymax), cex.axis = 0.8,
           main = sprintf("%s\nclass %d", unname(model_labels[m]), k))
      abline(v = 0, col = "red", lwd = 2, lty = 2)
      mval <- mean(v)
      arrows(x0 = mval, y0 = ymax * 0.65, x1 = mval, y1 = ymax * 0.45,
             col = "blue", lwd = 2, length = 0.08)
      ## Always scientific notation, so panels spanning wildly different
      ## magnitudes (1e-18 for LDA vs 1e-3 for Firth+WLW) stay comparable at a
      ## glance instead of switching between fixed and exponential format.
      lab <- sprintf("mean=%.1e", mval)
      ## Keep the label inside the panel: pos = 3 centres it on x, so shift x
      ## in by half the string width whenever the mean sits near either edge.
      usr <- par("usr")
      half <- strwidth(lab, cex = 0.85) / 2
      x_lab <- min(max(mval, usr[1] + half * 1.05), usr[2] - half * 1.05)
      text(x_lab, ymax * 0.65, lab, col = "blue", pos = 3, cex = 0.85)
    }
  }
  mtext(sprintf("Class means forming a %s right triangle (sd = 1), M = %d replications",
                geom_name, n_reps), outer = TRUE, cex = 1.0, font = 2)
  invisible(dev.off())
  cat("Wrote", out_png, "\n")
  invisible(results)
}

for (gname in names(geometries)) run_geometry(gname, geometries[[gname]])
