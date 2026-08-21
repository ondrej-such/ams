# Probability estimates for versicolor vs. virginica in the iris data,
# one panel per variate, comparing the same model set as two_class_prob.R.
# "Bayes (true params)" is replaced by a Gaussian plug-in (per-class normal,
# variances estimated separately = 1-D QDA), since iris has no known
# generative parameters.

suppressPackageStartupMessages({
  library(MASS)    # lda
  library(glmnet)  # L1/L2-penalized logistic regression
  library(logistf) # Firth's penalized-likelihood logistic regression
})

set.seed(1)

## ---- Data ----------------------------------------------------------------
# Class pair to compare; `pos` is the positive class.
neg <- "setosa"
pos <- "virginica"

d <- droplevels(subset(iris, Species %in% c(neg, pos)))
vars <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")
pair <- paste0(neg, "_", pos)

## ---- Fit every model for one variate, return probs on a grid -------------
fit_variate <- function(x, y, xg) {
  yb <- as.integer(y == pos)
  Np <- sum(yb); Nn <- sum(1 - yb)

  # Gaussian plug-in (per-class normal, separate variances, equal priors)
  mA <- mean(x[yb == 0]); sA <- sd(x[yb == 0])
  mB <- mean(x[yb == 1]); sB <- sd(x[yb == 1])
  fB <- dnorm(xg, mB, sB); fA <- dnorm(xg, mA, sA)
  p_gauss <- fB / (fA + fB)

  # LDA
  lda_fit <- lda(y ~ x)
  p_lda <- predict(lda_fit, data.frame(x = xg))$posterior[, pos]

  # L2 / L1 penalized logistic (glmnet needs >= 2 cols; standardize + pad)
  xm <- mean(x); xsd <- sd(x)
  X  <- cbind((x - xm) / xsd, 0)
  Xg <- cbind((xg - xm) / xsd, 0)
  cv2 <- cv.glmnet(X, y, family = "binomial", alpha = 0, standardize = FALSE)
  cv1 <- cv.glmnet(X, y, family = "binomial", alpha = 1, standardize = FALSE)
  p_l2 <- as.numeric(predict(cv2, newx = Xg, s = "lambda.min", type = "response"))
  p_l1 <- as.numeric(predict(cv1, newx = Xg, s = "lambda.min", type = "response"))

  # Firth
  ff <- logistf(yb ~ x)
  p_firth <- plogis(coef(ff)[1] + coef(ff)[2] * xg)

  # Platt scaling: standard and 2N-target variant
  t_std <- ifelse(yb == 1, (Np + 1) / (Np + 2), 1 / (Nn + 2))
  t_2N  <- ifelse(yb == 1, (2 * Np + 1) / (2 * Np + 2), 1 / (2 * Nn + 2))
  p_platt  <- predict(suppressWarnings(glm(t_std ~ x, family = binomial)),
                      data.frame(x = xg), type = "response")
  p_platt2 <- predict(suppressWarnings(glm(t_2N ~ x, family = binomial)),
                      data.frame(x = xg), type = "response")

  list(p_gauss, p_lda, p_l2, p_l1, p_firth, p_platt, p_platt2)
}

## ---- Model styling (shared) ----------------------------------------------
models <- data.frame(
  label = c("Gaussian (per-class normal)", "LDA", "L2 logistic",
            "L1 logistic", "Firth logistic", "Platt scaling",
            "Platt (2N targets)"),
  col   = c("#7570b3", "#1b9e77", "#d95f02", "#66a61e", "#e7298a",
            "#e6ab02", "#a6761d"),
  lty   = c(1, 1, 2, 4, 1, 5, 6),
  stringsAsFactors = FALSE
)

## ---- Draw one variate (own plot, with legend) ----------------------------
draw_one <- function(v) {
  x <- d[[v]]; y <- d$Species
  xg <- seq(min(x), max(x), length.out = 400)
  curves <- fit_variate(x, y, xg)

  par(mar = c(4, 4, 3, 1))
  plot(NA, xlim = range(xg), ylim = c(0, 1),
       xlab = v, ylab = paste0("P(", pos, " | x)"),
       main = paste0("iris: ", neg, " vs. ", pos, "  —  ", v))
  for (i in seq_len(nrow(models)))
    lines(xg, curves[[i]], lwd = 2, col = models$col[i], lty = models$lty[i])
  rug(x[y != pos], side = 1, col = "#1b9e7799")            # versicolor
  points(x[y == pos], rep(1, sum(y == pos)), pch = "|",    # virginica
         col = "#d95f0299")
  abline(h = 0.5, col = "grey70", lty = 3)
  legend("topleft", bty = "n", legend = models$label, col = models$col,
         lwd = 2, lty = models$lty, cex = 0.8)
}

dir <- getwd()  # write into the project directory (where this script is run from)
files <- setNames(
  file.path(dir, paste0("iris_", pair, "_", tolower(vars), ".png")), vars)

for (v in vars) {
  png(files[[v]], width = 7, height = 5, units = "in", res = 150)
  draw_one(v)
  invisible(dev.off())
  cat("Wrote", files[[v]], "\n")
}

# also a combined PDF (one variate per page)
pdf_out <- file.path(dir, paste0("iris_", pair, "_models.pdf"))
pdf(pdf_out, width = 7, height = 5)
for (v in vars) draw_one(v)
invisible(dev.off())
cat("Wrote", pdf_out, "\n")
