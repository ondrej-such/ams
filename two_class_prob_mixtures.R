# Branch of two_class_prob.R.
#
# Same two-class model-comparison machinery (Bayes-optimal rule vs. LDA,
# L2/L1-penalized logistic regression, Firth logistic regression, and two
# Platt-scaling variants), but run once for each of three class-conditional
# families sharing the same location parameters (0 and 12) and scale (1):
#   1. Mixture of two Cauchy distributions
#   2. Mixture of two Student-t distributions (df = 3)
#   3. Mixture of two Normal distributions        (same as two_class_prob.R)
#
# The point is to see how each fitted model's log-odds curve reacts as the
# class-conditional tails get heavier (Normal -> t(3) -> Cauchy), while the
# Bayes-optimal curve is computed from the TRUE generating density in each
# case.

suppressPackageStartupMessages({
  library(MASS)    # lda
  library(glmnet)  # L1/L2-penalized logistic regression
  library(logistf) # Firth's penalized-likelihood logistic regression
})

set.seed(1)

## ---- Shared settings -------------------------------------------------
n    <- 50
mu1  <- 0
mu2  <- 12
scl  <- 1     # scale parameter: sd for Normal, scale for Cauchy/t
df_t <- 3     # degrees of freedom for the Student-t family

models <- data.frame(
  label = c("Bayes (true params)", "LDA", "L2 logistic", "L1 logistic",
            "Firth logistic", "Platt scaling", "Platt (2N targets)",
            "Logistic (Brier score)"),
  col   = c("#7570b3", "#1b9e77", "#d95f02", "#66a61e", "#e7298a",
            "#e6ab02", "#a6761d", "#377eb8"),
  lty   = c(1, 1, 2, 4, 1, 5, 6, 3),
  stringsAsFactors = FALSE
)

lo <- function(p) qlogis(pmin(pmax(p, 1e-15), 1 - 1e-15))
ylim <- c(-10, 10)                        # window around the decision boundary
xg   <- seq(-10, 23, length.out = 400)    # fixed grid, same for all 3

## ---- Fit every model for one (x1, x2) sample, return a draw() closure ----
fit_and_plot <- function(x1, x2, dA, dB, family_label, file_stub) {
  x <- c(x1, x2)
  y <- factor(rep(c("A", "B"), each = n))
  dat <- data.frame(x = x, y = y)

  ## LDA
  lda_fit <- lda(y ~ x, data = dat)

  ## Penalized logistic regression (glmnet needs >= 2 cols; standardize + pad)
  x_mean <- mean(x); x_sd <- sd(x)
  xs <- (x - x_mean) / x_sd
  X  <- cbind(xs, 0)
  cv_l2 <- cv.glmnet(X, y, family = "binomial", alpha = 0, standardize = FALSE)
  cv_l1 <- cv.glmnet(X, y, family = "binomial", alpha = 1, standardize = FALSE)

  ## Firth logistic regression
  firth_fit <- logistf(y ~ x, data = data.frame(x = x, y = as.integer(y == "B")))
  fb0 <- coef(firth_fit)[1]; fb1 <- coef(firth_fit)[2]

  ## Platt scaling (standard and 2N-target variant)
  Np <- sum(y == "B"); Nn <- sum(y == "A")
  t_plus  <- (Np + 1) / (Np + 2);      t_minus  <- 1 / (Nn + 2)
  t_plus2 <- (2 * Np + 1) / (2 * Np + 2); t_minus2 <- 1 / (2 * Nn + 2)
  t_target  <- ifelse(y == "B", t_plus,  t_minus)
  t_target2 <- ifelse(y == "B", t_plus2, t_minus2)
  platt_fit  <- suppressWarnings(glm(t_target  ~ x, family = binomial))
  platt_fit2 <- suppressWarnings(glm(t_target2 ~ x, family = binomial))

  ## Logistic model fit by minimizing Brier score (mean squared error
  ## between predicted probability and the 0/1 label) instead of the usual
  ## log-likelihood, keeping the same logistic link p(x) = plogis(b0+b1*x).
  yb <- as.integer(y == "B")
  brier_loss <- function(b) mean((yb - plogis(b[1] + b[2] * x))^2)
  brier_fit  <- optim(c(0, 0), brier_loss, method = "BFGS")
  bb0 <- brier_fit$par[1]; bb1 <- brier_fit$par[2]

  ## Predictions on the shared grid
  p_lda    <- predict(lda_fit, newdata = data.frame(x = xg))$posterior[, "B"]
  Xg       <- cbind((xg - x_mean) / x_sd, 0)
  p_l2     <- as.numeric(predict(cv_l2, newx = Xg, s = "lambda.min", type = "response"))
  p_l1     <- as.numeric(predict(cv_l1, newx = Xg, s = "lambda.min", type = "response"))
  p_firth  <- plogis(fb0 + fb1 * xg)
  p_platt  <- predict(platt_fit,  newdata = data.frame(x = xg), type = "response")
  p_platt2 <- predict(platt_fit2, newdata = data.frame(x = xg), type = "response")
  p_brier  <- plogis(bb0 + bb1 * xg)

  ## Bayes-optimal, using the TRUE class-conditional densities
  fA <- dA(xg); fB <- dB(xg)
  p_bayes <- fB / (fA + fB)

  curves <- list(p_bayes, p_lda, p_l2, p_l1, p_firth, p_platt, p_platt2, p_brier)

  draw <- function() {
    par(mar = c(4, 4.5, 3, 1))
    plot(NA, xlim = range(xg), ylim = ylim,
         xlab = "x", ylab = "log-odds   log[ P(B|x) / P(A|x) ]",
         main = paste0("Class-B log-odds: fitted models vs. Bayes optimum  —  ",
                        family_label))
    for (i in seq_len(nrow(models)))
      lines(xg, lo(curves[[i]]), lwd = 2, col = models$col[i], lty = models$lty[i])

    rug(x1, side = 1, col = "#1b9e7799")                   # class A ticks
    points(x2, rep(ylim[2] - 0.4, n), pch = "|", col = "#d95f0299")  # class B

    abline(h = 0, col = "grey70", lty = 3)                 # decision boundary
    legend("topleft", bty = "n", legend = models$label,
           col = models$col, lwd = 2, lty = models$lty, cex = 0.85)
  }

  dir <- path.expand("~/ams")
  pdf_out <- file.path(dir, paste0(file_stub, ".pdf"))
  pdf(pdf_out, width = 7, height = 5); draw(); invisible(dev.off())
  cat("Wrote", pdf_out, "\n")

  png_out <- file.path(dir, paste0(file_stub, ".png"))
  png(png_out, width = 7, height = 5, units = "in", res = 150); draw(); invisible(dev.off())
  cat("Wrote", png_out, "\n")

  draw   # return the closure so it can be reused in the combined PDF below
}

## ---- 1. Mixture of Cauchy distributions -------------------------------
x1_c <- rcauchy(n, location = mu1, scale = scl)
x2_c <- rcauchy(n, location = mu2, scale = scl)
draw_cauchy <- fit_and_plot(
  x1_c, x2_c,
  dA = function(g) dcauchy(g, location = mu1, scale = scl),
  dB = function(g) dcauchy(g, location = mu2, scale = scl),
  family_label = "Cauchy mixture",
  file_stub = "two_class_prob_cauchy"
)

## ---- 2. Mixture of Student-t distributions (df = 3) --------------------
x1_t <- mu1 + scl * rt(n, df = df_t)
x2_t <- mu2 + scl * rt(n, df = df_t)
draw_t <- fit_and_plot(
  x1_t, x2_t,
  dA = function(g) dt((g - mu1) / scl, df = df_t) / scl,
  dB = function(g) dt((g - mu2) / scl, df = df_t) / scl,
  family_label = paste0("Student-t mixture (df = ", df_t, ")"),
  file_stub = "two_class_prob_t3"
)

## ---- 3. Mixture of Normal distributions ---------------------------------
x1_n <- rnorm(n, mean = mu1, sd = scl)
x2_n <- rnorm(n, mean = mu2, sd = scl)
draw_normal <- fit_and_plot(
  x1_n, x2_n,
  dA = function(g) dnorm(g, mean = mu1, sd = scl),
  dB = function(g) dnorm(g, mean = mu2, sd = scl),
  family_label = "Normal mixture",
  file_stub = "two_class_prob_normal"
)

## ---- Combined multi-page PDF (one family per page) ----------------------
combined_pdf <- file.path(path.expand("~/ams"), "two_class_prob_mixtures_all.pdf")
pdf(combined_pdf, width = 7, height = 5)
draw_cauchy(); draw_t(); draw_normal()
invisible(dev.off())
cat("Wrote", combined_pdf, "\n")
