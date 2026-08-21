# Probability estimates for two-class classification from a single variate.
# Two normal classes, n = 50 each, means 12 apart, sd = 1.
# Compare the Bayes-optimal rule against several fitted models:
#   LDA, L2- and L1-penalized logistic regression, Firth logistic
#   regression, and Platt scaling (sigmoid fit with regularized targets).

suppressPackageStartupMessages({
  library(MASS)    # lda
  library(glmnet)  # L1/L2-penalized logistic regression
  library(logistf) # Firth's penalized-likelihood logistic regression
})

set.seed(1)

## ---- Data ----------------------------------------------------------------
n   <- 50
mu1 <- 0
mu2 <- 12
sdv <- 1

x1 <- rnorm(n, mean = mu1, sd = sdv)
x2 <- rnorm(n, mean = mu2, sd = sdv)

x <- c(x1, x2)
y <- factor(rep(c("A", "B"), each = n))   # class B is the "positive" class

dat <- data.frame(x = x, y = y)

## ---- LDA -----------------------------------------------------------------
lda_fit <- lda(y ~ x, data = dat)

## ---- Penalized logistic regression (glmnet) ------------------------------
# glmnet needs >= 2 predictor columns; standardize x ourselves so the
# penalty acts on a scaled coefficient, and pad with a zero column.
x_mean <- mean(x); x_sd <- sd(x)
xs <- (x - x_mean) / x_sd
X  <- cbind(xs, 0)

# L2 (ridge, alpha = 0) and L1 (lasso, alpha = 1); lambda by CV.
cv_l2 <- cv.glmnet(X, y, family = "binomial", alpha = 0, standardize = FALSE)
cv_l1 <- cv.glmnet(X, y, family = "binomial", alpha = 1, standardize = FALSE)

## ---- Firth logistic regression -------------------------------------------
firth_fit <- logistf(y ~ x, data = data.frame(x = x, y = as.integer(y == "B")))
fb0 <- coef(firth_fit)[1]; fb1 <- coef(firth_fit)[2]

## ---- Platt scaling -------------------------------------------------------
# Platt (1999): fit a sigmoid 1/(1+exp(A*s + B)) to a score s by maximum
# likelihood against regularized soft targets, avoiding the overconfidence
# of hard 0/1 labels. Here the score is the variate x itself.
Np <- sum(y == "B"); Nn <- sum(y == "A")
t_plus  <- (Np + 1) / (Np + 2)
t_minus <- 1 / (Nn + 2)
t_target <- ifelse(y == "B", t_plus, t_minus)
platt_fit <- suppressWarnings(
  glm(t_target ~ x, family = binomial)   # soft targets => non-integer successes
)

# Variant: halve the pseudocount -> targets (2N+1)/(2N+2) and 1/(2N+2).
# These sit closer to 0/1, so the fitted sigmoid is steeper (less shrinkage).
t_plus2  <- (2 * Np + 1) / (2 * Np + 2)
t_minus2 <- 1 / (2 * Nn + 2)
t_target2 <- ifelse(y == "B", t_plus2, t_minus2)
platt_fit2 <- suppressWarnings(glm(t_target2 ~ x, family = binomial))

## ---- Prediction grid -----------------------------------------------------
xg <- seq(min(x) - 1, max(x) + 1, length.out = 400)

# P(class B | x) from LDA
p_lda <- predict(lda_fit, newdata = data.frame(x = xg))$posterior[, "B"]

# P(class B | x) from L2 / L1 penalized logistic regression
Xg <- cbind((xg - x_mean) / x_sd, 0)
p_l2 <- as.numeric(predict(cv_l2, newx = Xg, s = "lambda.min", type = "response"))
p_l1 <- as.numeric(predict(cv_l1, newx = Xg, s = "lambda.min", type = "response"))

# P(class B | x) from Firth logistic regression
p_firth <- plogis(fb0 + fb1 * xg)

# P(class B | x) from Platt scaling (standard and 2N-target variant)
p_platt  <- predict(platt_fit,  newdata = data.frame(x = xg), type = "response")
p_platt2 <- predict(platt_fit2, newdata = data.frame(x = xg), type = "response")

# P(class B | x) from the Bayes (optimal) classifier, using the TRUE
# means/variances and equal priors (0.5 each).
fA    <- dnorm(xg, mean = mu1, sd = sdv)
fB    <- dnorm(xg, mean = mu2, sd = sdv)
p_bayes <- fB / (fA + fB)

## ---- Plot ----------------------------------------------------------------
models <- data.frame(
  label = c("Bayes (true params)", "LDA", "L2 logistic", "L1 logistic",
            "Firth logistic", "Platt scaling", "Platt (2N targets)"),
  col   = c("#7570b3", "#1b9e77", "#d95f02", "#66a61e", "#e7298a",
            "#e6ab02", "#a6761d"),
  lty   = c(1, 1, 2, 4, 1, 5, 6),
  stringsAsFactors = FALSE
)
curves <- list(p_bayes, p_lda, p_l2, p_l1, p_firth, p_platt, p_platt2)

# log-odds transform; clip away from 0/1 so exact 0/1 don't map to +/-Inf.
lo <- function(p) qlogis(pmin(pmax(p, 1e-15), 1 - 1e-15))
ylim <- c(-10, 10)   # window around the decision boundary (log-odds = 0)

draw <- function() {
  par(mar = c(4, 4.5, 3, 1))
  plot(NA, xlim = range(xg), ylim = ylim,
       xlab = "x", ylab = "log-odds   log[ P(B|x) / P(A|x) ]",
       main = "Class-B log-odds: fitted models vs. Bayes optimum")
  for (i in seq_len(nrow(models)))
    lines(xg, lo(curves[[i]]), lwd = 2, col = models$col[i], lty = models$lty[i])

  # data as rugs (class A bottom, class B top of the window)
  rug(x1, side = 1, col = "#1b9e7799")
  points(x2, rep(ylim[2] - 0.4, n), pch = "|", col = "#d95f0299")

  abline(h = 0, col = "grey70", lty = 3)   # decision boundary

  legend("topleft", bty = "n", legend = models$label,
         col = models$col, lwd = 2, lty = models$lty, cex = 0.85)
}

dir <- getwd()  # write into the project directory (where this script is run from)

pdf_out <- file.path(dir, "two_class_prob.pdf")
pdf(pdf_out, width = 7, height = 5); draw(); invisible(dev.off())
cat("Wrote", pdf_out, "\n")

png_out <- file.path(dir, "two_class_prob.png")
png(png_out, width = 7, height = 5, units = "in", res = 150); draw(); invisible(dev.off())
cat("Wrote", png_out, "\n")
