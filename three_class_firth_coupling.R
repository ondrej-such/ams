# Complex experiment: 3-class classification using pairwise Firth logistic
# regression combined via Wu, Lin & Weng's (JMLR 2004) pairwise-coupling
# algorithm -- the same fixed-point iteration libsvm uses to turn one-vs-one
# SVM outputs into multi-class probability estimates.
#
# Setup
# -----
# Three classes, each bivariate Normal with identity covariance (sd = 1 in
# each dimension). The class means form a right triangle with sides
# 9, 12, 15 (the 3-4-5 triple scaled by 3: 9^2 + 12^2 = 15^2 = 225), with
# the right angle placed at class 1's mean:
#     |mu1 - mu2| =  9
#     |mu1 - mu3| = 12
#     |mu2 - mu3| = 15  (hypotenuse)
# All plots use a fixed 1:1 aspect ratio (asp = 1) so the right angle and
# true relative distances actually look correct on screen.
#
# For every pair of classes we fit a Firth (bias-reduced) logistic
# regression on the 2-D coordinates -- generalizing the 1-D Firth fit used
# earlier in this project to an arbitrary design matrix. The three pairwise
# fits are then combined at every point into a single 3-class probability
# vector two different ways: Wu-Lin-Weng (2004) coupling, and the earlier
# Hastie & Tibshirani (1998) coupling algorithm that WLW improved on. We
# also compute the exact Bayes-optimal classifier from the TRUE generating
# means (equal priors, identity covariance -> nearest-mean rule), and draw
# its decision boundaries on top of both fitted methods for comparison. We
# sample a uniform grid over the rectangle bounding the three means, couple
# predictions at every grid point, and visualize the result.

suppressPackageStartupMessages(library(MASS))   # lda() -- native multi-class comparison

set.seed(1)

## ---- 1. Class means: right triangle with sides 9, 12, 15 ------------------
mu1 <- c(0, 0)
mu2 <- c(9, 0)
mu3 <- c(0, 12)   # right angle at mu1; |mu2-mu3| = sqrt(9^2+12^2) = 15

means <- rbind(mu1, mu2, mu3)
rownames(means) <- paste0("class", 1:3)
cat("Class means:\n"); print(means)
cat(sprintf("Pairwise distances: |1-2|=%.3f  |1-3|=%.3f  |2-3|=%.3f\n\n",
            dist(rbind(mu1, mu2)), dist(rbind(mu1, mu3)), dist(rbind(mu2, mu3))))

## ---- 2. Sample data ---------------------------------------------------------
n <- 150   # points per class

sample_class <- function(mu, n) cbind(x = rnorm(n, mu[1], 1), y = rnorm(n, mu[2], 1))

X1 <- sample_class(mu1, n)
X2 <- sample_class(mu2, n)
X3 <- sample_class(mu3, n)

dat <- data.frame(
  x = c(X1[, 1], X2[, 1], X3[, 1]),
  y = c(X1[, 2], X2[, 2], X3[, 2]),
  class = factor(rep(1:3, each = n))
)

## ---- 3. Firth logistic regression, generalized to a p-column design -------
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
  yp <- as.integer(sub$class == hi)         # y = 1 for the higher-indexed class
  fit <- firth_logistic(Xp, yp)
  pair_fits[[paste0(lo, "_", hi)]] <- fit
  cat(sprintf("Firth pair (%d,%d): beta = (%.4f, %.4f, %.4f), iters = %d\n",
              lo, hi, fit$beta[1], fit$beta[2], fit$beta[3], fit$n_iter))
}
cat("\n")

## Pr(class a | class a or b) at points Xnew, for any a != b
pr_pair <- function(a, b, Xnew) {
  lo <- min(a, b); hi <- max(a, b)
  fit  <- pair_fits[[paste0(lo, "_", hi)]]
  p_hi <- predict_firth(fit, Xnew)          # Pr(class hi | a or b)
  if (a == hi) p_hi else 1 - p_hi
}

## ---- 4b. Fit one (2-class) LDA model per class pair ------------------------
## Same pairwise construction as the Firth fits above, but with LDA as the
## base binary classifier instead -- so LDA also goes through the coupling
## step rather than being fit natively on all 3 classes at once.
pair_fits_lda <- list()
for (pr in pairs) {
  lo <- pr[1]; hi <- pr[2]
  sub <- droplevels(dat[dat$class %in% c(lo, hi), ])
  pair_fits_lda[[paste0(lo, "_", hi)]] <- lda(class ~ x + y, data = sub)
}

## Pr(class a | class a or b) at points Xnew, for any a != b, via LDA
pr_pair_lda <- function(a, b, Xnew) {
  lo <- min(a, b); hi <- max(a, b)
  fit  <- pair_fits_lda[[paste0(lo, "_", hi)]]
  post <- predict(fit, newdata = data.frame(x = Xnew[, 1], y = Xnew[, 2]))$posterior
  p_hi <- post[, as.character(hi)]          # Pr(class hi | a or b)
  if (a == hi) p_hi else 1 - p_hi
}

## ---- 5. Wu, Lin & Weng (2004) pairwise coupling ----------------------------
## r: k x k matrix, r[i,j] = Pr(class i | class i or j) for i != j.
## Returns p (length k, sums to 1). Same fixed-point iteration used in
## libsvm's multiclass_probability().
wlw_couple <- function(r, max_iter = 200, eps = NULL) {
  k <- nrow(r)
  if (is.null(eps)) eps <- 0.005 / k
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

## ---- 5b. Hastie & Tibshirani (1998) pairwise coupling ----------------------
## The original pairwise-coupling algorithm (WLW's "Method 1" / their point
## of comparison): find p minimizing the weighted KL divergence between
## r_ij and mu_ij = p_i/(p_i+p_j) -- equivalent to fitting a Bradley-Terry
## paired-comparison model by iterative proportional scaling. All pairwise
## Firth fits here use equal training-set sizes (n_ij constant), so the
## n_ij weights cancel and can be dropped.
ht_couple <- function(r, max_iter = 200, tol = 1e-8) {
  k <- nrow(r)
  p <- rep(1 / k, k)
  for (iter in 1:max_iter) {
    p_old <- p
    for (i in 1:k) {
      mu_i <- p[i] / (p[i] + p[-i])   # mu_ij for j != i, using current p
      p[i] <- p[i] * sum(r[i, -i]) / sum(mu_i)
      p <- p / sum(p)
    }
    if (max(abs(p - p_old)) < tol) break
  }
  p
}

## With classes this well separated (9-15 units apart, sd = 1), some
## pairwise fits underflow to a probability of EXACTLY 0 or 1 in double
## precision. That makes Q[t,t] = sum(r[-t,t]^2) exactly 0 in wlw_couple,
## a divide-by-zero -> NaN -> the comparison in the while loop breaks. Clamp
## away from the boundary so every r_ij stays strictly in (0, 1).
clamp01 <- function(p, eps = 1e-10) pmin(pmax(p, eps), 1 - eps)

## Couple 3 pairwise predictions into a 3-class probability matrix, one row
## per point in Xnew (an n x 2 matrix of x,y coordinates). `couple_fn` picks
## the coupling algorithm (wlw_couple / ht_couple); `pr_pair_fn` picks the
## base pairwise classifier (pr_pair for Firth / pr_pair_lda for LDA).
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

## ---- 5c. Exact Bayes-optimal classifier from the TRUE generating model ----
## Equal priors, identity covariance, known means -> Bayes rule is exactly
## the nearest-mean classifier; posterior = softmax(-0.5 * squared distance).
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

## ---- 5d. LDA, fit natively on all 3 classes at once (no coupling needed) --
## Unlike logistic/Firth, LDA's discriminant functions generalize directly
## to k > 2 classes, so this is a genuine one-shot multi-class fit, kept
## alongside the pairwise LDA+coupling versions below for comparison.
lda_fit <- lda(class ~ x + y, data = dat)
lda_predict <- function(Xnew) {
  predict(lda_fit, newdata = data.frame(x = Xnew[, 1], y = Xnew[, 2]))$posterior
}

## ---- 6. Validate on the training data --------------------------------------
Xtr <- as.matrix(dat[, c("x", "y")])
acc <- function(P) mean(max.col(P) == as.integer(dat$class))

Ptr_wlw     <- couple_predict(Xtr, wlw_couple, pr_pair)
Ptr_ht      <- couple_predict(Xtr, ht_couple,  pr_pair)
Ptr_lda_wlw <- couple_predict(Xtr, wlw_couple, pr_pair_lda)
Ptr_lda_ht  <- couple_predict(Xtr, ht_couple,  pr_pair_lda)
Ptr_lda     <- lda_predict(Xtr)
Ptr_bayes   <- bayes_predict(Xtr)

cat("Training-set confusion matrix (Firth + Wu-Lin-Weng coupling):\n")
print(table(true = dat$class, pred = max.col(Ptr_wlw)))
cat("Training-set confusion matrix (Firth + Hastie-Tibshirani coupling):\n")
print(table(true = dat$class, pred = max.col(Ptr_ht)))
cat("Training-set confusion matrix (LDA + Wu-Lin-Weng coupling):\n")
print(table(true = dat$class, pred = max.col(Ptr_lda_wlw)))
cat("Training-set confusion matrix (LDA + Hastie-Tibshirani coupling):\n")
print(table(true = dat$class, pred = max.col(Ptr_lda_ht)))
cat("Training-set confusion matrix (LDA, native multi-class):\n")
print(table(true = dat$class, pred = max.col(Ptr_lda)))
cat("Training-set confusion matrix (true Bayes-optimal rule):\n")
print(table(true = dat$class, pred = max.col(Ptr_bayes)))
cat(sprintf(paste0("\nTraining accuracy -- Firth+WLW: %.1f%%   Firth+HT: %.1f%%   ",
                    "LDA+WLW: %.1f%%   LDA+HT: %.1f%%   LDA(native): %.1f%%   Bayes: %.1f%%\n\n"),
            100 * acc(Ptr_wlw), 100 * acc(Ptr_ht),
            100 * acc(Ptr_lda_wlw), 100 * acc(Ptr_lda_ht),
            100 * acc(Ptr_lda), 100 * acc(Ptr_bayes)))

## ---- 7. Grid over the rectangle bounding the three means -------------------
margin <- 4
gx <- seq(min(means[, 1]) - margin, max(means[, 1]) + margin, length.out = 100)
gy <- seq(min(means[, 2]) - margin, max(means[, 2]) + margin, length.out = 100)
grid <- expand.grid(x = gx, y = gy)   # x varies fastest
Xg <- as.matrix(grid)

P         <- couple_predict(Xg, wlw_couple, pr_pair)
P_ht      <- couple_predict(Xg, ht_couple,  pr_pair)
P_lda_wlw <- couple_predict(Xg, wlw_couple, pr_pair_lda)
P_lda_ht  <- couple_predict(Xg, ht_couple,  pr_pair_lda)
P_lda     <- lda_predict(Xg)
P_bayes   <- bayes_predict(Xg)

grid$p1 <- P[, 1]; grid$p2 <- P[, 2]; grid$p3 <- P[, 3]
grid$pred <- max.col(P)
grid$pred_ht <- max.col(P_ht)
grid$pred_lda_wlw <- max.col(P_lda_wlw)
grid$pred_lda_ht <- max.col(P_lda_ht)
grid$pred_lda <- max.col(P_lda)
grid$pred_bayes <- max.col(P_bayes)

nx <- length(gx); ny <- length(gy)

## Overlay the true Bayes decision boundaries (as a dashed black line) on
## the current plot. Works by contouring the Bayes argmax field at the two
## half-integer levels that trace the edges between all three regions.
add_bayes_boundary <- function() {
  contour(gx, gy, matrix(grid$pred_bayes, nx, ny), levels = c(1.5, 2.5),
          add = TRUE, drawlabels = FALSE, lwd = 2, lty = 2, col = "black")
}

## ---- 8. Visualize -----------------------------------------------------------
dir <- path.expand(".")
class_cols <- c("#e41a1c", "#4daf4a", "#377eb8")   # red, green, blue

## 8a. RGB probability blend: class1=red, class2=green, class3=blue
Rm <- matrix(grid$p1, nx, ny)
Gm <- matrix(grid$p2, nx, ny)
Bm <- matrix(grid$p3, nx, ny)
rgb_arr <- array(0, dim = c(ny, nx, 3))
rgb_arr[, , 1] <- t(Rm)[ny:1, ]
rgb_arr[, , 2] <- t(Gm)[ny:1, ]
rgb_arr[, , 3] <- t(Bm)[ny:1, ]
rgb_raster <- as.raster(rgb_arr)
print(file.path(dir, "three_class_firth_coupling_rgb.png"))
png(file.path(dir, "three_class_firth_coupling_rgb.png"),
    width = 7, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))
plot(NA, xlim = range(gx), ylim = range(gy), xlab = "x", ylab = "y", asp = 1,
     main = "3-class Firth + Wu-Lin-Weng coupling\n(R = class1, G = class2, B = class3)")
rasterImage(rgb_raster, min(gx), min(gy), max(gx), max(gy))
points(dat$x, dat$y, col = class_cols[dat$class], pch = 16, cex = 0.5)
points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)
add_bayes_boundary()
legend("topright", bty = "n", lty = 2, lwd = 2, col = "black", legend = "Bayes boundary")
invisible(dev.off())
cat("Wrote", file.path(dir, "three_class_firth_coupling_rgb.png"), "\n")

## 8b. Hard (argmax) decision regions
print(file.path(dir, "three_class_firth_coupling_argmax.png"))
png(file.path(dir, "three_class_firth_coupling_argmax.png"),
    width = 7, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))
image(gx, gy, matrix(grid$pred, nx, ny),
      col = adjustcolor(class_cols, alpha.f = 0.35), breaks = c(0.5, 1.5, 2.5, 3.5),
      xlab = "x", ylab = "y", asp = 1,
      main = "3-class Firth + Wu-Lin-Weng coupling: argmax regions")
points(dat$x, dat$y, col = class_cols[dat$class], pch = 16, cex = 0.5)
points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)
add_bayes_boundary()
legend("topright", bty = "n", legend = c(paste("class", 1:3), "Bayes boundary"),
       col = c(class_cols, "black"), pch = c(16, 16, 16, NA), lty = c(NA, NA, NA, 2), lwd = 2)
invisible(dev.off())
cat("Wrote", file.path(dir, "three_class_firth_coupling_argmax.png"), "\n")

## 8c. Individual coupled-probability heatmaps, one per class
png(file.path(dir, "three_class_firth_coupling_probs.png"),
    width = 13, height = 4.3, units = "in", res = 150)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
for (k in 1:3) {
  zk <- matrix(P[, k], nx, ny)
  image(gx, gy, zk, zlim = c(0, 1), col = hcl.colors(50, "YlOrRd", rev = TRUE),
        xlab = "x", ylab = "y", asp = 1, main = paste0("P(class ", k, " | x)  [coupled, WLW]"))
  points(dat$x, dat$y, pch = 16, cex = 0.3,
         col = adjustcolor(class_cols[dat$class], alpha.f = 0.5))
  contour(gx, gy, zk, levels = 0.5, add = TRUE, lwd = 2)
  add_bayes_boundary()
}
invisible(dev.off())
cat("Wrote", file.path(dir, "three_class_firth_coupling_probs.png"), "\n")

## 8d. RGB probability blend using Hastie-Tibshirani coupling instead of WLW
Rm_ht <- matrix(P_ht[, 1], nx, ny)
Gm_ht <- matrix(P_ht[, 2], nx, ny)
Bm_ht <- matrix(P_ht[, 3], nx, ny)
rgb_arr_ht <- array(0, dim = c(ny, nx, 3))
rgb_arr_ht[, , 1] <- t(Rm_ht)[ny:1, ]
rgb_arr_ht[, , 2] <- t(Gm_ht)[ny:1, ]
rgb_arr_ht[, , 3] <- t(Bm_ht)[ny:1, ]
rgb_raster_ht <- as.raster(rgb_arr_ht)

png(file.path(dir, "three_class_ht_coupling_rgb.png"),
    width = 7, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))
plot(NA, xlim = range(gx), ylim = range(gy), xlab = "x", ylab = "y", asp = 1,
     main = "3-class Firth + Hastie-Tibshirani coupling\n(R = class1, G = class2, B = class3)")
rasterImage(rgb_raster_ht, min(gx), min(gy), max(gx), max(gy))
points(dat$x, dat$y, col = class_cols[dat$class], pch = 16, cex = 0.5)
points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)
add_bayes_boundary()
legend("topright", bty = "n", lty = 2, lwd = 2, col = "black", legend = "Bayes boundary")
invisible(dev.off())
cat("Wrote", file.path(dir, "three_class_ht_coupling_rgb.png"), "\n")

## 8e. Hastie-Tibshirani argmax decision regions
png(file.path(dir, "three_class_ht_coupling_argmax.png"),
    width = 7, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))
image(gx, gy, matrix(grid$pred_ht, nx, ny),
      col = adjustcolor(class_cols, alpha.f = 0.35), breaks = c(0.5, 1.5, 2.5, 3.5),
      xlab = "x", ylab = "y", asp = 1,
      main = "3-class Firth + Hastie-Tibshirani coupling: argmax regions")
points(dat$x, dat$y, col = class_cols[dat$class], pch = 16, cex = 0.5)
points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)
add_bayes_boundary()
legend("topright", bty = "n", legend = c(paste("class", 1:3), "Bayes boundary"),
       col = c(class_cols, "black"), pch = c(16, 16, 16, NA), lty = c(NA, NA, NA, 2), lwd = 2)
invisible(dev.off())
cat("Wrote", file.path(dir, "three_class_ht_coupling_argmax.png"), "\n")

## 8f. RGB probability blend using LDA (native multi-class, no coupling)
Rm_lda <- matrix(P_lda[, 1], nx, ny)
Gm_lda <- matrix(P_lda[, 2], nx, ny)
Bm_lda <- matrix(P_lda[, 3], nx, ny)
rgb_arr_lda <- array(0, dim = c(ny, nx, 3))
rgb_arr_lda[, , 1] <- t(Rm_lda)[ny:1, ]
rgb_arr_lda[, , 2] <- t(Gm_lda)[ny:1, ]
rgb_arr_lda[, , 3] <- t(Bm_lda)[ny:1, ]
rgb_raster_lda <- as.raster(rgb_arr_lda)

png(file.path(dir, "three_class_lda_rgb.png"),
    width = 7, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))
plot(NA, xlim = range(gx), ylim = range(gy), xlab = "x", ylab = "y", asp = 1,
     main = "3-class LDA (native multi-class)\n(R = class1, G = class2, B = class3)")
rasterImage(rgb_raster_lda, min(gx), min(gy), max(gx), max(gy))
points(dat$x, dat$y, col = class_cols[dat$class], pch = 16, cex = 0.5)
points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)
add_bayes_boundary()
legend("topright", bty = "n", lty = 2, lwd = 2, col = "black", legend = "Bayes boundary")
invisible(dev.off())
cat("Wrote", file.path(dir, "three_class_lda_rgb.png"), "\n")

## 8g. LDA argmax decision regions
png(file.path(dir, "three_class_lda_argmax.png"),
    width = 7, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))
image(gx, gy, matrix(grid$pred_lda, nx, ny),
      col = adjustcolor(class_cols, alpha.f = 0.35), breaks = c(0.5, 1.5, 2.5, 3.5),
      xlab = "x", ylab = "y", asp = 1,
      main = "3-class LDA: argmax regions")
points(dat$x, dat$y, col = class_cols[dat$class], pch = 16, cex = 0.5)
points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)
add_bayes_boundary()
legend("topright", bty = "n", legend = c(paste("class", 1:3), "Bayes boundary"),
       col = c(class_cols, "black"), pch = c(16, 16, 16, NA), lty = c(NA, NA, NA, 2), lwd = 2)
invisible(dev.off())
cat("Wrote", file.path(dir, "three_class_lda_argmax.png"), "\n")

## ---- Helper to draw one RGB-blend / argmax pair for a given coupled fit ---
plot_coupled <- function(P_k, pred_k, method_label, file_stub) {
  Rm_k <- matrix(P_k[, 1], nx, ny)
  Gm_k <- matrix(P_k[, 2], nx, ny)
  Bm_k <- matrix(P_k[, 3], nx, ny)
  rgb_arr_k <- array(0, dim = c(ny, nx, 3))
  rgb_arr_k[, , 1] <- t(Rm_k)[ny:1, ]
  rgb_arr_k[, , 2] <- t(Gm_k)[ny:1, ]
  rgb_arr_k[, , 3] <- t(Bm_k)[ny:1, ]
  rgb_raster_k <- as.raster(rgb_arr_k)

  png(file.path(dir, paste0(file_stub, "_rgb.png")),
      width = 7, height = 6, units = "in", res = 150)
  par(mar = c(4, 4, 3, 1))
  plot(NA, xlim = range(gx), ylim = range(gy), xlab = "x", ylab = "y", asp = 1,
       main = paste0("3-class ", method_label,
                      "\n(R = class1, G = class2, B = class3)"))
  rasterImage(rgb_raster_k, min(gx), min(gy), max(gx), max(gy))
  points(dat$x, dat$y, col = class_cols[dat$class], pch = 16, cex = 0.5)
  points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)
  add_bayes_boundary()
  legend("topright", bty = "n", lty = 2, lwd = 2, col = "black", legend = "Bayes boundary")
  invisible(dev.off())
  cat("Wrote", file.path(dir, paste0(file_stub, "_rgb.png")), "\n")

  png(file.path(dir, paste0(file_stub, "_argmax.png")),
      width = 7, height = 6, units = "in", res = 150)
  par(mar = c(4, 4, 3, 1))
  image(gx, gy, matrix(pred_k, nx, ny),
        col = adjustcolor(class_cols, alpha.f = 0.35), breaks = c(0.5, 1.5, 2.5, 3.5),
        xlab = "x", ylab = "y", asp = 1,
        main = paste0("3-class ", method_label, ": argmax regions"))
  points(dat$x, dat$y, col = class_cols[dat$class], pch = 16, cex = 0.5)
  points(means[, 1], means[, 2], pch = 4, lwd = 3, cex = 1.5)
  add_bayes_boundary()
  legend("topright", bty = "n", legend = c(paste("class", 1:3), "Bayes boundary"),
         col = c(class_cols, "black"), pch = c(16, 16, 16, NA), lty = c(NA, NA, NA, 2), lwd = 2)
  invisible(dev.off())
  cat("Wrote", file.path(dir, paste0(file_stub, "_argmax.png")), "\n")
}

## 8h. LDA + Wu-Lin-Weng coupling
plot_coupled(P_lda_wlw, grid$pred_lda_wlw, "LDA + Wu-Lin-Weng coupling",
             "three_class_lda_wlw_coupling")

## 8i. LDA + Hastie-Tibshirani coupling
plot_coupled(P_lda_ht, grid$pred_lda_ht, "LDA + Hastie-Tibshirani coupling",
             "three_class_lda_ht_coupling")
