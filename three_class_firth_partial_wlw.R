# Firth logistic regression + PARTIAL Wu-Lin-Weng coupling.
#
# Same 3-class setup as three_class_firth_coupling.R (self-contained here):
# bivariate Normal classes, identity covariance, means forming a 9-12-15
# right triangle (right angle at class 1), with a pairwise Firth logistic
# fit for each of the 3 class pairs.
#
# The full WLW (2004) coupling finds p minimizing the least-squares
# objective built from ALL THREE pairwise terms:
#     L(p) = sum_{i<j} (r_ji*p_i - r_ij*p_j)^2      over (1,2), (1,3), (2,3)
#
# Here we instead build the objective from only TWO of those three terms
# -- "partial" coupling -- dropping one pairwise comparison entirely from
# the optimization (though all 3 Firth models are still fit; the dropped
# pair's classifier just never enters the coupling objective). There are
# 3 ways to pick 2 of 3 terms, i.e. 3 ways to choose which pair to drop:
#     drop (1,2): keep (1,3) and (2,3)
#     drop (1,3): keep (1,2) and (2,3)
#     drop (2,3): keep (1,2) and (1,3)
# We solve all three with the same fixed-point iteration WLW uses (it's a
# general quadratic-minimization-on-the-simplex solver; it doesn't care how
# Q was built), plot argmax decision regions for each, and overlay the true
# Bayes boundary plus the full (3-term) WLW result for reference.

suppressPackageStartupMessages(library(MASS))

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

## ---- 3. Firth logistic regression (multi-dim design matrix) ---------------
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
  list(beta = beta, n_iter = i)
}
predict_firth <- function(fit, Xnew) plogis(as.vector(cbind(1, Xnew) %*% fit$beta))

## ---- 4. Fit one Firth model per class pair ---------------------------------
pairs <- combn(3, 2, simplify = FALSE)   # (1,2) (1,3) (2,3)
pair_fits <- list()
for (pr in pairs) {
  lo <- pr[1]; hi <- pr[2]
  sub <- dat[dat$class %in% c(lo, hi), ]
  Xp <- as.matrix(sub[, c("x", "y")])
  yp <- as.integer(sub$class == hi)
  fit <- firth_logistic(Xp, yp)
  pair_fits[[paste0(lo, "_", hi)]] <- fit
  cat(sprintf("Firth pair (%d,%d): beta = (%.4f, %.4f, %.4f), iters = %d\n",
              lo, hi, fit$beta[1], fit$beta[2], fit$beta[3], fit$n_iter))
}
cat("\n")

pr_pair <- function(a, b, Xnew) {
  lo <- min(a, b); hi <- max(a, b)
  fit  <- pair_fits[[paste0(lo, "_", hi)]]
  p_hi <- predict_firth(fit, Xnew)
  if (a == hi) p_hi else 1 - p_hi
}

## ---- 5. Wu-Lin-Weng coupling, generalized to an arbitrary subset of pairs -
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
wlw_drop12 <- function(r) wlw_couple_general(r, list(c(1, 3), c(2, 3)))  # drop pair (1,2)
wlw_drop13 <- function(r) wlw_couple_general(r, list(c(1, 2), c(2, 3)))  # drop pair (1,3)
wlw_drop23 <- function(r) wlw_couple_general(r, list(c(1, 2), c(1, 3)))  # drop pair (2,3)

## Clamp pairwise probabilities away from exactly 0/1 (well-separated Firth
## fits can underflow there, which would make some Q[t,t] exactly 0).
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

## ---- 6. Exact Bayes-optimal classifier from the TRUE generating model -----
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

## ---- 7. Validate on the training data --------------------------------------
Xtr <- as.matrix(dat[, c("x", "y")])
acc <- function(P) mean(max.col(P) == as.integer(dat$class))

Ptr_full   <- couple_predict(Xtr, wlw_full)
Ptr_drop12 <- couple_predict(Xtr, wlw_drop12)
Ptr_drop13 <- couple_predict(Xtr, wlw_drop13)
Ptr_drop23 <- couple_predict(Xtr, wlw_drop23)
Ptr_bayes  <- bayes_predict(Xtr)

cat(sprintf(paste0("Training accuracy -- full WLW: %.1f%%   drop(1,2): %.1f%%   ",
                    "drop(1,3): %.1f%%   drop(2,3): %.1f%%   Bayes: %.1f%%\n\n"),
            100 * acc(Ptr_full), 100 * acc(Ptr_drop12),
            100 * acc(Ptr_drop13), 100 * acc(Ptr_drop23), 100 * acc(Ptr_bayes)))

## ---- 8. Grid over the rectangle bounding the three means -------------------
margin <- 4
gx <- seq(min(means[, 1]) - margin, max(means[, 1]) + margin, length.out = 100)
gy <- seq(min(means[, 2]) - margin, max(means[, 2]) + margin, length.out = 100)
grid <- expand.grid(x = gx, y = gy)
Xg <- as.matrix(grid)
nx <- length(gx); ny <- length(gy)

grid$pred_full   <- max.col(couple_predict(Xg, wlw_full))
grid$pred_drop12 <- max.col(couple_predict(Xg, wlw_drop12))
grid$pred_drop13 <- max.col(couple_predict(Xg, wlw_drop13))
grid$pred_drop23 <- max.col(couple_predict(Xg, wlw_drop23))
grid$pred_bayes  <- max.col(bayes_predict(Xg))

add_bayes_boundary <- function() {
  contour(gx, gy, matrix(grid$pred_bayes, nx, ny), levels = c(1.5, 2.5),
          add = TRUE, drawlabels = FALSE, lwd = 2, lty = 2, col = "black")
}

## ---- 9. Visualize: argmax regions for the 3 partial-coupling variants -----
## (plus the full 3-term WLW coupling as a reference panel)
dir <- path.expand("~/ams")
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
            "Firth + partial WLW coupling\n(objective drops pair 1-2, keeps 1-3 & 2-3)",
            "three_class_firth_partial_wlw_drop12_argmax.png")
plot_argmax(grid$pred_drop13,
            "Firth + partial WLW coupling\n(objective drops pair 1-3, keeps 1-2 & 2-3)",
            "three_class_firth_partial_wlw_drop13_argmax.png")
plot_argmax(grid$pred_drop23,
            "Firth + partial WLW coupling\n(objective drops pair 2-3, keeps 1-2 & 1-3)",
            "three_class_firth_partial_wlw_drop23_argmax.png")
plot_argmax(grid$pred_full,
            "Firth + full (3-term) WLW coupling  [reference]",
            "three_class_firth_full_wlw_argmax_reference.png")
