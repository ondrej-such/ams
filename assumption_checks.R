# -----------------------------------------------------------------------------
# assumption_checks.R
#
# The three properties claimed for the reduced data in the Partial one-vs-one
# section, each measured on the English (training) and the Urdu (test)
# narration in the same discriminant plane:
#
#   1. bivariate normality within each speaker  -- Royston's multivariate
#      Shapiro-Wilk test
#   2. a common covariance across the three     -- Box's M test
#   3. how far apart the classes are            -- Cohen's d, which for two
#      classes with a shared covariance equals the Mahalanobis distance
#
# Base R only, so a collaborator needs no packages beyond a standard install.
# Both tests are ports of the standard implementations and have been checked
# against them: Royston against MVN::royston, Box's M against heplots::boxM,
# agreeing to the printed precision.
#
# The plane is fitted on the TRAINING cell only, so the test cell is an
# unbiased check; the training cell is reported alongside for completeness.
# -----------------------------------------------------------------------------

source("data_preparation.R")   # leaves `tr` (english) and `te` (urdu) in memory

spk <- sort(unique(tr$speaker))
Xtr192 <- as.matrix(tr[, -1]); ytr <- factor(tr$speaker, levels = spk)
Xte192 <- as.matrix(te[, -1]); yte <- factor(te$speaker, levels = spk)
p192 <- ncol(Xtr192); K3 <- 3

## ---- the same regularized LDA projection used for the figure ---------------

alpha3 <- 0.5
nk3    <- as.integer(table(ytr))
mu192  <- t(sapply(spk, function(k) colMeans(Xtr192[ytr == k, ])))
Sw192  <- Reduce(`+`, lapply(1:K3, function(i)
            (nk3[i] - 1) * cov(Xtr192[ytr == spk[i], ]))) / (nrow(Xtr192) - K3)
Sw192  <- (1 - alpha3) * Sw192 + alpha3 * mean(diag(Sw192)) * diag(p192)
ew192  <- eigen(Sw192, symmetric = TRUE)
Wh192  <- ew192$vectors %*% diag(1 / sqrt(pmax(ew192$values, 1e-12))) %*% t(ew192$vectors)
V2     <- eigen(cov(mu192 %*% Wh192) * (K3 - 1), symmetric = TRUE)$vectors[, 1:2]
P2     <- Wh192 %*% V2

cells <- list(
  list(lab = "english (training)", Z = Xtr192 %*% P2, g = ytr),
  list(lab = "urdu (test)",        Z = Xte192 %*% P2, g = yte)
)

## ---- 1. Royston's multivariate Shapiro-Wilk --------------------------------
# Shapiro-Wilk is univariate. Royston's H transforms the per-coordinate W
# statistics to normality, sums them, and rescales by an equivalent degrees of
# freedom that accounts for the correlation between the coordinates -- so it
# tests joint normality rather than the margins separately. Valid for 12 <= n
# <= 2000; Shapiro-Francia replaces Shapiro-Wilk on heavy-tailed coordinates,
# as in Royston's original.

royston_H <- function(X) {
  X <- as.matrix(X); n <- nrow(X); p <- ncol(X)
  if (n < 12 || n > 2000) stop("royston_H is implemented for 12 <= n <= 2000")

  kurt <- function(v) { m <- mean(v); mean((v - m)^4) / mean((v - m)^2)^2 }
  sf_W <- function(v) cor(sort(v), qnorm(ppoints(length(v), a = 3/8)))^2

  lx <- log(n)
  mu <- -1.5861 - 0.31082 * lx - 0.083751 * lx^2 + 0.0038915 * lx^3
  sg <- exp(-0.4803 - 0.082676 * lx + 0.0030302 * lx^2)

  z <- vapply(seq_len(p), function(i) {
    v <- X[, i]
    W <- if (kurt(v) > 3) sf_W(v) else as.numeric(shapiro.test(v)$statistic)
    (log(1 - W) - mu) / sg
  }, numeric(1))

  u <- 0.715; v <- 0.21364 + 0.015124 * lx^2 - 0.0018034 * lx^3; l <- 5
  C  <- cor(X)
  NC <- (C^l) * (1 - (u * (1 - C)^u) / v)
  mC  <- (sum(NC) - p) / (p^2 - p)
  edf <- p / (1 + (p - 1) * mC)
  H   <- edf * sum(qnorm(pnorm(-z) / 2)^2) / p
  c(H = H, p.value = pchisq(H, df = edf, lower.tail = FALSE))
}

cat("\n===== 1. Royston's multivariate Shapiro-Wilk, per speaker =====\n")
cat(sprintf("%-20s %-8s %6s %10s %10s\n", "cell", "speaker", "n", "H", "p"))
for (cl in cells) for (k in spk) {
  Zi <- cl$Z[cl$g == k, ]
  r  <- royston_H(Zi)
  cat(sprintf("%-20s %-8s %6d %10.3f %10.4f\n", cl$lab, k, nrow(Zi),
              r["H"], r["p.value"]))
}

## ---- 2. Box's M for a common covariance ------------------------------------
# Chi-square approximation with Box's correction factor.

box_M <- function(Z, g) {
  lv <- levels(droplevels(as.factor(g)))
  gq <- length(lv); m <- nrow(Z); q <- ncol(Z)
  ni <- as.integer(table(g)[lv])
  Si <- lapply(lv, function(k) cov(Z[g == k, , drop = FALSE]))
  Sp <- Reduce(`+`, Map(function(S, a) (a - 1) * S, Si, ni)) / (m - gq)
  M  <- (m - gq) * log(det(Sp)) - sum((ni - 1) * sapply(Si, function(S) log(det(S))))
  c1 <- (sum(1 / (ni - 1)) - 1 / (m - gq)) *
        (2 * q^2 + 3 * q - 1) / (6 * (q + 1) * (gq - 1))
  df <- (gq - 1) * q * (q + 1) / 2
  c(chisq = M * (1 - c1), df = df,
    p.value = pchisq(M * (1 - c1), df, lower.tail = FALSE))
}

cat("\n===== 2. Box's M for equality of the three covariances =====\n")
cat(sprintf("%-20s %10s %8s %10s\n", "cell", "chi-sq", "df", "p"))
for (cl in cells) {
  b <- box_M(cl$Z, cl$g)
  cat(sprintf("%-20s %10.3f %8d %10.4f\n", cl$lab,
              b["chisq"], as.integer(b["df"]), b["p.value"]))
}

## ---- 3. Cohen's d between each pair of speakers ----------------------------
# For two classes sharing a covariance, Cohen's d along the Fisher axis IS the
# Mahalanobis distance. In-sample D^2 is upward biased, so the Lachenbruch
# correction is applied; both values are reported.

cohens_d <- function(Z, g, a, b) {
  i <- g == a; j <- g == b; k <- ncol(Z); n1 <- sum(i); n2 <- sum(j)
  S  <- ((n1 - 1) * cov(Z[i, ]) + (n2 - 1) * cov(Z[j, ])) / (n1 + n2 - 2)
  dm <- colMeans(Z[i, ]) - colMeans(Z[j, ])
  d2 <- as.numeric(t(dm) %*% solve(S) %*% dm)
  c(raw = sqrt(d2),
    adj = sqrt(max((n1 + n2 - k - 3) / (n1 + n2 - 2) * d2 - k * (1/n1 + 1/n2), 0)))
}

cat("\n===== 3. Cohen's d between speakers =====\n")
cat(sprintf("%-20s %-6s %10s %12s %12s\n",
            "cell", "pair", "Cohen d", "corrected", "overlap"))
for (cl in cells) for (pr in combn(spk, 2, simplify = FALSE)) {
  d <- cohens_d(cl$Z, cl$g, pr[1], pr[2])
  cat(sprintf("%-20s %s-%s %10.2f %12.2f %11.1f%%\n", cl$lab, pr[1], pr[2],
              d["raw"], d["adj"], 200 * pnorm(-d["adj"] / 2)))
}
cat("\noverlap is the share of each distribution falling on the wrong side of\n")
cat("the optimal boundary, 2*pnorm(-d/2), under equal-variance normality.\n")
