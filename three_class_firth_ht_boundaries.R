# Firth logistic regression + Hastie-Tibshirani coupling: decision BOUNDARIES.
#
# Same 3-class setup as the other three_class_* scripts (self-contained here):
# bivariate Normal classes, identity covariance, means forming a 9-12-15 right
# triangle (right angle at class 1), 150 points per class.
#
# Where three_class_firth_coupling.R renders coupled predictions as filled
# argmax regions, this script draws the decision BOUNDARIES themselves as
# curves, so the three rules can be overlaid in a single panel and compared
# directly:
#   - Hastie & Tibshirani (1998) coupling  [the subject of this script]
#   - Wu, Lin & Weng (2004) coupling       [for comparison]
#   - the exact Bayes-optimal rule from the true generating means
#
# A note on partial (2-of-3-term) coupling, by analogy with
# three_class_firth_partial_wlw.R: there is no point running that experiment
# with HT. With pair (i,j) dropped, the two kept pairs (i,k) and (j,k) share
# class k as a hub, and BOTH objectives -- WLW's least squares and HT's
# weighted KL divergence -- attain their global minimum of exactly zero at the
# same point, namely p_i : p_j : p_k = r_ik/r_ki : r_jk/r_kj : 1. So partial
# HT and partial WLW are the same estimator and produce identical figures.
# The two couplings can only differ when all three terms are present, since
# then the system is generally overdetermined and no exact solution exists --
# which is precisely the case this script visualizes.

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
predict_firth_eta <- function(fit, Xnew) as.vector(cbind(1, Xnew) %*% fit$beta)

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

## Signed log-odds of class a over class b -- positive favors a.
log_odds <- function(a, b, Xnew) {
  lo <- min(a, b); hi <- max(a, b)
  eta <- predict_firth_eta(pair_fits[[paste0(lo, "_", hi)]], Xnew)
  if (a == hi) eta else -eta
}
pr_pair <- function(a, b, Xnew) plogis(log_odds(a, b, Xnew))

## ---- 5a. Hastie & Tibshirani (1998) coupling -------------------------------
## Find p minimizing the weighted KL divergence between r_ij and
## mu_ij = p_i/(p_i+p_j) -- equivalently, fit a Bradley-Terry paired-comparison
## model by iterative proportional scaling. All pairwise fits here use equal
## training-set sizes, so the n_ij weights cancel and are dropped.
## tol matched to wlw_couple's tightened eps below, so the disagreement count
## reflects a real difference between the two couplings rather than one of them
## simply being stopped earlier than the other.
ht_couple <- function(r, max_iter = 500, tol = 1e-12) {
  k <- nrow(r)
  p <- rep(1 / k, k)
  for (iter in 1:max_iter) {
    p_old <- p
    for (i in 1:k) {
      mu_i <- p[i] / (p[i] + p[-i])
      p[i] <- p[i] * sum(r[i, -i]) / sum(mu_i)
      p <- p / sum(p)
    }
    if (max(abs(p - p_old)) < tol) break
  }
  p
}

## ---- 5b. Wu, Lin & Weng (2004) coupling, for comparison --------------------
## eps tightened from WLW's published default of 0.005/k: that default is an
## absolute tolerance on max|Qp - pQp| and leaves ~1e-4 of error in log10(p)
## at moderate log-odds. Harmless for filled argmax regions, but this script
## reports the PERCENTAGE of grid points where HT and WLW disagree, and near a
## decision boundary the two top classes are nearly tied, so a loose tolerance
## can flip the argmax and inflate that count.
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

## Clamp pairwise probabilities away from exactly 0/1: with classes this well
## separated the Firth fits saturate in double precision, which would put a
## zero in HT's denominator and a zero on WLW's diagonal.
clamp01 <- function(p, eps = 1e-10) pmin(pmax(p, eps), 1 - eps)

couple_predict <- function(Xnew, couple_fn) {
  r12 <- clamp01(pr_pair(1, 2, Xnew)); r21 <- 1 - r12
  r13 <- clamp01(pr_pair(1, 3, Xnew)); r31 <- 1 - r13
  r23 <- clamp01(pr_pair(2, 3, Xnew)); r32 <- 1 - r23
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
cat(sprintf("Training accuracy -- Firth+HT: %.1f%%   Firth+WLW: %.1f%%   Bayes: %.1f%%\n\n",
            100 * acc(couple_predict(Xtr, ht_couple)),
            100 * acc(couple_predict(Xtr, wlw_couple)),
            100 * acc(bayes_predict(Xtr))))

## ---- 8. Grid --------------------------------------------------------------
## Finer than the argmax-region scripts (200 vs 100 per axis): we are drawing
## boundary curves by contouring the argmax field, so grid resolution sets how
## smooth and how accurately placed those curves are.
margin <- 4
gx <- seq(min(means[, 1]) - margin, max(means[, 1]) + margin, length.out = 200)
gy <- seq(min(means[, 2]) - margin, max(means[, 2]) + margin, length.out = 200)
grid <- expand.grid(x = gx, y = gy)
Xg <- as.matrix(grid)
nx <- length(gx); ny <- length(gy)

pred_ht    <- max.col(couple_predict(Xg, ht_couple))
pred_wlw   <- max.col(couple_predict(Xg, wlw_couple))
pred_bayes <- max.col(bayes_predict(Xg))

cat(sprintf("Grid points where HT and WLW disagree: %d of %d (%.2f%%)\n\n",
            sum(pred_ht != pred_wlw), length(pred_ht),
            100 * mean(pred_ht != pred_wlw)))

## Trace the edges between all three argmax regions by contouring the argmax
## field at the two half-integer levels.
add_boundary <- function(pred, col, lty, lwd = 2) {
  contour(gx, gy, matrix(pred, nx, ny), levels = c(1.5, 2.5),
          add = TRUE, drawlabels = FALSE, col = col, lty = lty, lwd = lwd)
}

## ---- 9. Plot ---------------------------------------------------------------
dir <- path.expand("~/ams")
class_cols <- c("#e41a1c", "#4daf4a", "#377eb8")   # red, green, blue
col_ht  <- "#d95f02"
col_wlw <- "#7570b3"

png(file.path(dir, "three_class_firth_ht_boundaries.png"),
    width = 7, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))
plot(NA, xlim = range(gx), ylim = range(gy), asp = 1, xlab = "x", ylab = "y",
     main = "3-class Firth logistic regression: decision boundaries\nHastie-Tibshirani vs. Wu-Lin-Weng coupling")

## Training data, faint, for context.
points(dat$x, dat$y, col = adjustcolor(class_cols[dat$class], alpha.f = 0.35),
       pch = 16, cex = 0.5)
points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)

add_boundary(pred_bayes, "black",  2)   # dashed
add_boundary(pred_wlw,   col_wlw,  3)   # dotted
add_boundary(pred_ht,    col_ht,   1)   # solid

legend("topright", bty = "n",
       legend = c("Firth + Hastie-Tibshirani", "Firth + Wu-Lin-Weng", "Bayes-optimal"),
       col = c(col_ht, col_wlw, "black"), lty = c(1, 3, 2), lwd = 2)
invisible(dev.off())
cat("Wrote", file.path(dir, "three_class_firth_ht_boundaries.png"), "\n")
