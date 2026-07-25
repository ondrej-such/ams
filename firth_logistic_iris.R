## Firth (bias-reduced) logistic regression, fit by hand via iterative
## penalized Newton-Raphson (IRLS with Jeffreys-prior correction).
##
## Data: iris, setosa (y=0) vs virginica (y=1), predictor = Petal.Length.
## Motivation: this predictor perfectly separates the two species, so
## ordinary ML logistic regression has no finite MLE (coefficients diverge).
## Firth's penalized likelihood removes the first-order bias and gives a
## finite, well-behaved estimate even under separation.

set.seed(1)

## ---- 1. Data ---------------------------------------------------------
df <- iris[iris$Species %in% c("setosa", "virginica"), ]
df$Species <- factor(df$Species, levels = c("setosa", "virginica"))
y <- as.numeric(df$Species) - 1              # 0 = setosa, 1 = virginica
x <- df$Petal.Length
X <- cbind(Intercept = 1, Petal.Length = x)   # n x 2 design matrix
n <- nrow(X); p <- ncol(X)

cat(sprintf("n = %d, separable check: setosa max = %.2f, virginica min = %.2f\n\n",
            n, max(x[y == 0]), min(x[y == 1])))

## ---- 2. Firth penalized log-likelihood (for tracking only) ----------
penlogLik <- function(beta) {
  eta <- as.vector(X %*% beta)
  p_i <- plogis(eta)
  W <- p_i * (1 - p_i)
  XtWX <- t(X) %*% (X * W)
  ll <- sum(y * eta - log(1 + exp(eta)))
  0.5 * as.numeric(determinant(XtWX, logarithm = TRUE)$modulus) + ll
}

## ---- 3. Iterative fit: penalized Newton-Raphson (Firth 1993) --------
beta <- c(0, 0)                 # starting values
tol  <- 1e-8
max_iter <- 25

cat(sprintf("%-4s %12s %12s %14s %12s\n",
            "iter", "beta0", "beta1", "pen.logLik", "max|delta|"))
cat(sprintf("%-4d %12.6f %12.6f %14.6f %12s\n",
            0, beta[1], beta[2], penlogLik(beta), "-"))

converged <- FALSE
for (iter in 1:max_iter) {
  eta <- as.vector(X %*% beta)
  p_i <- plogis(eta)
  W   <- p_i * (1 - p_i)

  XtWX     <- t(X) %*% (X * W)          # observed information
  XtWX_inv <- solve(XtWX)

  ## hat matrix diagonal h_i = W_i * x_i' (X'WX)^-1 x_i
  h <- W * rowSums((X %*% XtWX_inv) * X)

  ## Firth-modified score: adds h_i*(0.5 - p_i) correction to residual
  U_star <- t(X) %*% (y - p_i + h * (0.5 - p_i))

  step <- as.vector(XtWX_inv %*% U_star)
  beta_new <- beta + step

  delta <- max(abs(step))
  cat(sprintf("%-4d %12.6f %12.6f %14.6f %12.2e\n",
              iter, beta_new[1], beta_new[2], penlogLik(beta_new), delta))

  beta <- beta_new
  if (delta < tol) { converged <- TRUE; break }
}

cat(sprintf("\nConverged: %s after %d iterations\n", converged, iter))
cat(sprintf("Firth estimates:  beta0 (Intercept) = %.6f,  beta1 (Petal.Length) = %.6f\n",
            beta[1], beta[2]))

## ---- 4. Standard errors from final observed information -------------
eta <- as.vector(X %*% beta)
p_i <- plogis(eta)
W   <- p_i * (1 - p_i)
XtWX <- t(X) %*% (X * W)
se <- sqrt(diag(solve(XtWX)))
cat(sprintf("SE:               beta0 = %.6f,  beta1 = %.6f\n", se[1], se[2]))

## ---- 5. Cross-checks --------------------------------------------------
cat("\n--- Cross-check: ordinary glm() (expect separation / non-convergence) ---\n")
fit_glm <- suppressWarnings(glm(y ~ x, family = binomial))
print(coef(summary(fit_glm)))
cat(sprintf("glm() converged flag: %s\n", fit_glm$converged))

if (requireNamespace("logistf", quietly = TRUE)) {
  cat("\n--- Cross-check: logistf package ---\n")
  library(logistf)
  fit_firth <- logistf(y ~ x)
  print(coef(fit_firth))
} else {
  cat("\n(logistf package not installed - skipping package cross-check)\n")
}
