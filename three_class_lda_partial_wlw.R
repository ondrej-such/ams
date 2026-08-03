# LDA one-vs-one + PARTIAL Wu-Lin-Weng coupling.
#
# Same 3-class setup as three_class_firth_coupling.R / three_class_firth_partial_wlw.R
# (self-contained here): bivariate Normal classes, identity covariance, means
# forming a 9-12-15 right triangle (right angle at class 1), 150 points per
# class. The base pairwise (one-vs-one) classifier here is LDA instead of
# Firth logistic regression -- one MASS::lda fit per class pair, the same
# construction as pair_fits_lda / pr_pair_lda in three_class_firth_coupling.R
# -- combined with the full (3-term) Wu-Lin-Weng coupling and with the same
# three "partial" coupling variants (each dropping one pairwise term from the
# coupling objective) used in three_class_firth_partial_wlw.R. This is the
# LDA analogue of that script's four argmax figures, for a direct comparison
# of how much the choice of base one-vs-one classifier (Firth vs. LDA)
# matters for the partial-coupling behavior.

set.seed(1)

## ---- 1. Class means: right triangle with sides 9, 12, 15 ------------------
mu1 <- c(0, 0)
mu2 <- c(9, 0)
mu3 <- c(0, 12)   # right angle at mu1; |mu2-mu3| = sqrt(9^2+12^2) = 15
means <- rbind(mu1, mu2, mu3)
rownames(means) <- paste0("class", 1:3)

## ---- 2. Sample data ---------------------------------------------------------
n <- 150   # points per class
sample_class <- function(mu, n) cbind(x = rnorm(n, mu[1], 1), y = rnorm(n, mu[2], 1))
X1 <- sample_class(mu1, n); X2 <- sample_class(mu2, n); X3 <- sample_class(mu3, n)
dat <- data.frame(
  x = c(X1[, 1], X2[, 1], X3[, 1]),
  y = c(X1[, 2], X2[, 2], X3[, 2]),
  class = factor(rep(1:3, each = n))
)

## ---- 3. One-vs-one LDA, via the closed-form pooled-covariance discriminant
## (rather than MASS::lda + predict.lda's posterior), so the pairwise
## comparison is available as a raw linear log-odds score at full floating-
## point precision. predict.lda()$posterior rounds to *exactly* 1.0 (or 0.0)
## once the discriminant score exceeds about 37 in magnitude -- routine here,
## since the classes are 9-15 units apart with unit variance. That's
## harmless for the *full* 3-term coupling below (the other two pairwise
## terms keep each class's Q[t,t] away from zero regardless), but it
## silently destroys exactly the information the *partial*-coupling closed
## form needs: far from the hub class, deciding between the two dropped-pair
## classes comes down to the *difference* of two individually-huge log-odds,
## and that difference is a smooth, well-behaved quantity even where each
## log-odds on its own has long since saturated a raw probability to 0/1.
pairs <- combn(3, 2, simplify = FALSE)   # (1,2) (1,3) (2,3)
pair_coef <- list()
for (pr in pairs) {
  lo <- pr[1]; hi <- pr[2]
  Xlo <- as.matrix(dat[dat$class == lo, c("x", "y")])
  Xhi <- as.matrix(dat[dat$class == hi, c("x", "y")])
  n_lo <- nrow(Xlo); n_hi <- nrow(Xhi)
  mlo <- colMeans(Xlo); mhi <- colMeans(Xhi)
  Sigma <- ((n_lo - 1) * cov(Xlo) + (n_hi - 1) * cov(Xhi)) / (n_lo + n_hi - 2)
  Sinv  <- solve(Sigma)
  w <- as.numeric(Sinv %*% (mhi - mlo))
  b <- -0.5 * as.numeric(mhi %*% Sinv %*% mhi - mlo %*% Sinv %*% mlo) +
       log(n_hi / n_lo)
  pair_coef[[paste0(lo, "_", hi)]] <- list(w = w, b = b)
}

## Raw signed log-odds of class hi over class lo (lo < hi), at full
## precision -- positive favors hi. This is what MASS::lda's linear
## discriminant reduces to for two groups; computing it directly (instead of
## going through predict.lda) is what lets it stay informative however far
## Xnew is from the training data.
eta_pair_raw <- function(lo, hi, Xnew) {
  cf <- pair_coef[[paste0(lo, "_", hi)]]
  as.numeric(Xnew %*% cf$w + cf$b)
}

## Signed log-odds of class a over class b (a != b, either order) --
## positive favors a.
log_odds <- function(a, b, Xnew) {
  lo <- min(a, b); hi <- max(a, b)
  eta <- eta_pair_raw(lo, hi, Xnew)
  if (a == hi) eta else -eta
}

## Pr(class a | class a or b) at points Xnew, for any a != b -- used only by
## the full coupling below, where saturation is harmless.
pr_pair <- function(a, b, Xnew) plogis(log_odds(a, b, Xnew))

## ---- 4. Wu-Lin-Weng coupling, generalized to an arbitrary subset of pairs -
## `include`: a list of pairs, e.g. list(c(1,2), c(1,3)), that defines which
## terms of the least-squares objective L(p) = sum (r_ji p_i - r_ij p_j)^2
## are included. Passing all 3 pairs reproduces the original full WLW
## algorithm exactly; passing 2 gives the "partial" coupling used here.
wlw_couple_general <- function(r, include, max_iter = 200, eps = NULL) {
  k <- nrow(r)
  if (is.null(eps)) eps <- 0.005 / k
  Q <- matrix(0, k, k)
  for (pr in include) {
    a <- pr[1]; b <- pr[2]
    Q[a, a] <- Q[a, a] + r[b, a]^2
    Q[b, b] <- Q[b, b] + r[a, b]^2
    Q[a, b] <- Q[a, b] - r[b, a] * r[a, b]
    Q[b, a] <- Q[b, a] - r[a, b] * r[b, a]
  }
  p <- rep(1 / k, k)
  for (iter in 1:max_iter) {
    Qp  <- as.vector(Q %*% p)
    pQp <- sum(p * Qp)
    if (max(abs(Qp - pQp)) < eps) break
    for (t in 1:k) {
      if (Q[t, t] == 0) next   # class t not covered by any included pair
      diff  <- (-Qp[t] + pQp) / Q[t, t]
      p[t]  <- p[t] + diff
      pQp   <- (pQp + diff * (diff * Q[t, t] + 2 * Qp[t])) / (1 + diff)^2
      Qp    <- (Qp + diff * Q[t, ]) / (1 + diff)
      p     <- p / (1 + diff)
    }
  }
  p / sum(p)
}

all_pairs <- list(c(1, 2), c(1, 3), c(2, 3))
wlw_full   <- function(r) wlw_couple_general(r, all_pairs)

## Partial coupling (exactly 2 of the 3 pairs kept) has a closed form and
## doesn't need the iterative solver at all. With pair (i,j) dropped, the
## two kept pairs (i,k) and (j,k) share the third class k as a common "hub",
## so the objective splits into two INDEPENDENT squared terms -- one in
## (p_i, p_k), one in (p_j, p_k) -- each of which can be driven to exactly
## zero, giving p_i : p_j : p_k = exp(eta_ik) : exp(eta_jk) : 1, where
## eta_ik = log(p_i/p_k) is pair (i,k)'s signed log-odds. We compute this
## directly from the RAW log-odds (log_odds(), never saturated) via the
## standard softmax stabilization (subtract the max before exponentiating),
## rather than first collapsing each side to a probability and then
## dividing -- the latter is what produced the numerical noise before: once
## both r[i,k] and r[j,k] round to exactly 1.0, their ratio carries no
## information about which of i, j actually dominates.
couple_predict_partial <- function(Xnew, i, j, k) {
  eta_i <- log_odds(i, k, Xnew)   # positive favors i over k
  eta_j <- log_odds(j, k, Xnew)   # positive favors j over k
  Pm <- matrix(0, nrow(Xnew), 3)
  for (row in seq_len(nrow(Xnew))) {
    L <- numeric(3)
    L[i] <- eta_i[row]; L[j] <- eta_j[row]; L[k] <- 0
    w <- exp(L - max(L))
    Pm[row, ] <- w / sum(w)
  }
  Pm
}
couple_predict_drop12 <- function(Xnew) couple_predict_partial(Xnew, i = 1, j = 2, k = 3)
couple_predict_drop13 <- function(Xnew) couple_predict_partial(Xnew, i = 1, j = 3, k = 2)
couple_predict_drop23 <- function(Xnew) couple_predict_partial(Xnew, i = 2, j = 3, k = 1)

## Clamp pairwise probabilities away from exactly 0/1 (well-separated LDA
## fits can be numerically exactly 0/1, which would make some Q[t,t] exactly
## zero).
clamp01 <- function(p, eps = 1e-10) pmin(pmax(p, eps), 1 - eps)

couple_predict <- function(Xnew, couple_fn, pr_pair_fn = pr_pair) {
  r12 <- clamp01(pr_pair_fn(1, 2, Xnew)); r21 <- 1 - r12
  r13 <- clamp01(pr_pair_fn(1, 3, Xnew)); r31 <- 1 - r13
  r23 <- clamp01(pr_pair_fn(2, 3, Xnew)); r32 <- 1 - r23
  Pm <- matrix(0, nrow(Xnew), 3)
  for (i in seq_len(nrow(Xnew))) {
    r <- matrix(0, 3, 3)
    r[1, 2] <- r12[i]; r[2, 1] <- r21[i]
    r[1, 3] <- r13[i]; r[3, 1] <- r31[i]
    r[2, 3] <- r23[i]; r[3, 2] <- r32[i]
    Pm[i, ] <- couple_fn(r)
  }
  Pm
}

## ---- 5. Exact Bayes-optimal classifier from the TRUE generating model -----
bayes_predict <- function(Xnew) {
  D <- cbind(
    rowSums((Xnew - matrix(mu1, nrow(Xnew), 2, byrow = TRUE))^2),
    rowSums((Xnew - matrix(mu2, nrow(Xnew), 2, byrow = TRUE))^2),
    rowSums((Xnew - matrix(mu3, nrow(Xnew), 2, byrow = TRUE))^2)
  )
  logf <- -0.5 * D
  m <- apply(logf, 1, max)
  ex <- exp(logf - m)
  ex / rowSums(ex)
}

## ---- 6. Validate on the training data --------------------------------------
Xtr <- as.matrix(dat[, c("x", "y")])
acc <- function(P) mean(max.col(P) == as.integer(dat$class))

Ptr_full   <- couple_predict(Xtr, wlw_full)
Ptr_drop12 <- couple_predict_drop12(Xtr)
Ptr_drop13 <- couple_predict_drop13(Xtr)
Ptr_drop23 <- couple_predict_drop23(Xtr)
Ptr_bayes  <- bayes_predict(Xtr)

cat(sprintf(paste0("Training accuracy (LDA one-vs-one) -- full WLW: %.1f%%   drop(1,2): %.1f%%   ",
                    "drop(1,3): %.1f%%   drop(2,3): %.1f%%   Bayes: %.1f%%\n\n"),
            100 * acc(Ptr_full), 100 * acc(Ptr_drop12),
            100 * acc(Ptr_drop13), 100 * acc(Ptr_drop23), 100 * acc(Ptr_bayes)))

## ---- 7. Grid over the rectangle bounding the three means -------------------
margin <- 4
gx <- seq(min(means[, 1]) - margin, max(means[, 1]) + margin, length.out = 100)
gy <- seq(min(means[, 2]) - margin, max(means[, 2]) + margin, length.out = 100)
grid <- expand.grid(x = gx, y = gy)
Xg <- as.matrix(grid)
nx <- length(gx); ny <- length(gy)

grid$pred_full   <- max.col(couple_predict(Xg, wlw_full))
grid$pred_drop12 <- max.col(couple_predict_drop12(Xg))
grid$pred_drop13 <- max.col(couple_predict_drop13(Xg))
grid$pred_drop23 <- max.col(couple_predict_drop23(Xg))
grid$pred_bayes  <- max.col(bayes_predict(Xg))

add_bayes_boundary <- function() {
  contour(gx, gy, matrix(grid$pred_bayes, nx, ny), levels = c(1.5, 2.5),
          add = TRUE, drawlabels = FALSE, lwd = 2, lty = 2, col = "black")
}

## ---- 8. Visualize: argmax regions for the 3 partial-coupling variants -----
## (plus the full 3-term WLW coupling as a reference panel)
dir <- path.expand(".")
class_cols <- c("#e41a1c", "#4daf4a", "#377eb8")   # red, green, blue

plot_argmax <- function(pred_col, title, file_stub) {
  png(file.path(dir, file_stub), width = 7, height = 6, units = "in", res = 150)
  par(mar = c(4, 4, 3, 1))
  image(gx, gy, matrix(pred_col, nx, ny),
        col = adjustcolor(class_cols, alpha.f = 0.35), breaks = c(0.5, 1.5, 2.5, 3.5),
        xlab = "x", ylab = "y", asp = 1, main = title)
  points(dat$x, dat$y, col = class_cols[dat$class], pch = 16, cex = 0.5)
  points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)
  add_bayes_boundary()
  legend("topright", bty = "n", legend = c(paste("class", 1:3), "Bayes boundary"),
         col = c(class_cols, "black"), pch = c(16, 16, 16, NA), lty = c(NA, NA, NA, 2), lwd = 2)
  invisible(dev.off())
  cat("Wrote", file.path(dir, file_stub), "\n")
}

plot_argmax(grid$pred_drop12,
            "LDA one-vs-one + partial WLW coupling\n(objective drops pair 1-2, keeps 1-3 & 2-3)",
            "three_class_lda_partial_wlw_drop12_argmax.png")
plot_argmax(grid$pred_drop13,
            "LDA one-vs-one + partial WLW coupling\n(objective drops pair 1-3, keeps 1-2 & 2-3)",
            "three_class_lda_partial_wlw_drop13_argmax.png")
plot_argmax(grid$pred_drop23,
            "LDA one-vs-one + partial WLW coupling\n(objective drops pair 2-3, keeps 1-2 & 1-3)",
            "three_class_lda_partial_wlw_drop23_argmax.png")
plot_argmax(grid$pred_full,
            "LDA one-vs-one + full (3-term) WLW coupling  [reference]",
            "three_class_lda_full_wlw_argmax_reference.png")
