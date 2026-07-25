## Iterative self-training Firth logistic regression vs. LDA ground truth.
##
## Data: iris, setosa (y=0) vs virginica (y=1), predictor = Petal.Length.
##
## Every iteration fits FIRTH (bias-reduced) logistic regression via the
## same penalized Newton-Raphson used earlier (needed because Petal.Length
## perfectly separates the classes, so an ordinary glm() has no finite
## MLE). Iteration 1 fits on the TRUE labels. Iteration k >= 2 fits on the
## FITTED PROBABILITIES from iteration k-1 (a self-training / bootstrapping
## loop).
## At every iteration we print the coefficients and the cross-entropy
## between that iteration's fitted probabilities and the LDA classifier's
## posterior probability of "virginica", which we treat as ground truth.

suppressMessages(library(MASS))   # for lda()

## ---- 1. Data -----------------------------------------------------------
df <- iris[iris$Species %in% c("setosa", "virginica"), ]
df$Species <- factor(df$Species, levels = c("setosa", "virginica"))
y <- as.numeric(df$Species) - 1     # 0 = setosa, 1 = virginica
x <- df$Petal.Length

## ---- 2. Ground truth: LDA posterior P(virginica | x) -------------------
lda_fit  <- lda(Species ~ Petal.Length, data = df)
p_lda    <- predict(lda_fit)$posterior[, "virginica"]

## ---- 3. Cross-entropy helper -------------------------------------------
cross_entropy <- function(p_true, p_pred, eps = 1e-12) {
  p_pred <- pmin(pmax(p_pred, eps), 1 - eps)
  -mean(p_true * log(p_pred) + (1 - p_true) * log(1 - p_pred))
}

## ---- 4. Firth logistic regression (penalized Newton-Raphson) -----------
## Same algorithm as firth_logistic_iris.R, packaged as a function.
## Works for continuous response in [0,1] as well as binary 0/1: the
## Firth (Jeffreys-prior) correction term only depends on x, not y.
firth_logistic <- function(x, y, tol = 1e-8, max_iter = 25) {
  X <- cbind(Intercept = 1, x = x)
  beta <- c(0, 0)
  for (i in 1:max_iter) {
    eta  <- as.vector(X %*% beta)
    p_i  <- plogis(eta)
    W    <- p_i * (1 - p_i)
    XtWX <- t(X) %*% (X * W)
    XtWX_inv <- solve(XtWX)
    h <- W * rowSums((X %*% XtWX_inv) * X)
    U_star <- t(X) %*% (y - p_i + h * (0.5 - p_i))
    step <- as.vector(XtWX_inv %*% U_star)
    beta_new <- beta + step
    beta <- beta_new
    if (max(abs(step)) < tol) break
  }
  list(beta = beta, fitted = plogis(as.vector(X %*% beta)), n_iter = i)
}

## ---- 5. Iterative loop: Firth first, then glm on own predictions --------
n_iter <- 10000

cat(sprintf("%-4s %10s %12s %12s %14s %10s\n",
            "iter", "method", "beta0", "beta1", "crossent_LDA", "converged"))

## Iteration 1: Firth fit on the true labels
fit1  <- firth_logistic(x, y)
p_hat <- fit1$fitted
cat(sprintf("%-4d %10s %12.6f %12.6f %14.6f %10s\n",
            1, "firth", fit1$beta[1], fit1$beta[2],
            cross_entropy(p_lda, p_hat), "TRUE"))

## Iterations 2..n_iter: ordinary glm() on previous fitted probabilities
response <- p_hat
for (iter in 2:n_iter) {
  fit   <- suppressWarnings(glm(response ~ x, family = binomial))
  p_hat <- fitted(fit)
  ce    <- cross_entropy(p_lda, p_hat)

  cat(sprintf("%-4d %10s %12.6f %12.6f %14.6f %10s\n",
              iter, "glm", coef(fit)[1], coef(fit)[2], ce, fit$converged))

  response <- p_hat
}

print(coef(lda_fit))
