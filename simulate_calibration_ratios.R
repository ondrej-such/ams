# Monte Carlo calibration-ratio simulation.
#
# Repeatedly sample N = 50 points per class from two unit-variance Normal
# distributions with means 6 apart (class A ~ N(0,1), class B ~ N(6,1)),
# fit eight probability models -- L1-penalized logistic regression,
# L2-penalized logistic regression, Firth logistic regression, Platt
# scaling, Platt scaling with halved pseudocounts ("2N" targets), LDA,
# Bayesian logistic regression with Gelman et al.'s (2008) weakly
# informative Cauchy(0, 2.5) prior (posterior mode via arm::bayesglm), and
# the same with a Student-t (df = 4) prior -- and record, for every sampled
# point in every replication, the ratio
# of each model's PREDICTED probability to the TRUE Bayes-optimal
# probability of being class B, log_p_hat / log_p_true. The downstream
# analysis (index.qmd) restricts this to class-A points only: class A is
# centered 6 SDs away from class B, so for a class-A point p_true = P(B|x)
# is essentially always well away from 1 (it would take a > 3-SD draw from
# class A's own distribution to approach it), and the ratio never
# saturates the way it would for class-B points, where p_true ~ 1 for
# almost every sample. Since the whole setup -- both Gaussians and all six
# fitting procedures -- is symmetric under swapping the class labels and
# reflecting x around the midpoint, class-A and class-B behavior are
# mirror images in distribution, so nothing is lost by only looking at
# class A; it just keeps the metric away from the saturated regime without
# needing an ad hoc min(p, 1-p) transform. Each replication is also
# flagged for whether its two classes happen to be linearly separable in x
# (i.e. some threshold perfectly splits class A from class B) -- this is
# common when the two means are 6 SDs apart, and it is exactly the regime
# in which unregularized/weakly-regularized fits blow up toward 0/1 and can
# look "perfectly calibrated" by fluke.
#
# Everything is computed on the LOG-probability scale. Every model here
# reduces to a linear predictor (log-odds) eta(x) -- the true Bayes rule
# because it comes from two equal-variance Gaussians, and each fitted
# model because that's what logistic/LDA/Platt models are -- so instead of
# forming p_hat = plogis(eta) (which rounds to exactly 0 or 1 once |eta| is
# a couple hundred, forcing an ad hoc eps clamp before taking ratios/logs),
# we go straight from eta to log(p) via plogis(eta, log.p = TRUE). That
# function is implemented with log1p() internally and stays accurate however
# extreme eta gets, so no clamping is ever needed and log10(ratio) is just a
# difference of two well-behaved numbers.
#
# Output: a long-format CSV (rep, point_id, class, x, model, log_p_true,
# log_p_hat, log10_ratio, separable), one row per (replication, point,
# model) combination. log10_ratio = (log_p_hat - log_p_true) / log(10).

suppressPackageStartupMessages({
  library(glmnet)   # L1/L2-penalized logistic regression
  library(logistf)  # Firth's penalized-likelihood logistic regression
  library(arm)      # bayesglm: Gelman et al. weakly informative priors
})

set.seed(1)

n_reps <- 30    # Monte Carlo replications (M = 30)
n      <- 50    # points per class
mu1    <- 0
mu2    <- 6
sdv    <- 1
ln10   <- log(10)

fit_one_rep <- function(rep_id) {
  x1 <- rnorm(n, mu1, sdv)
  x2 <- rnorm(n, mu2, sdv)
  x  <- c(x1, x2)
  y  <- factor(rep(c("A", "B"), each = n))
  yb <- as.integer(y == "B")

  ## True Bayes-optimal log-probability at each sampled point. For two
  ## equal-variance Gaussians with equal priors, log(fB/fA) is exactly
  ## linear in x, so p_true(x) = plogis(a_true + b_true * x) -- same
  ## algebra as the two-Gaussian sigmoid discussed earlier in this project.
  b_true <- (mu2 - mu1) / sdv^2
  a_true <- -(mu2^2 - mu1^2) / (2 * sdv^2)
  eta_true  <- a_true + b_true * x
  log_p_true <- plogis(eta_true, log.p = TRUE)

  ## Is this replication's sample linearly separable in x, i.e. does some
  ## threshold perfectly split class A from class B?
  separable <- (max(x1) < min(x2)) || (max(x2) < min(x1))

  ## L1 / L2 penalized logistic regression (glmnet needs >= 2 predictor
  ## columns; standardize + pad with a zero column, as in two_class_prob.R).
  ## type = "link" returns the linear predictor (log-odds) directly, so we
  ## never form the linear-scale probability.
  x_mean <- mean(x); x_sd <- sd(x)
  X <- cbind((x - x_mean) / x_sd, 0)
  cv_l2 <- cv.glmnet(X, y, family = "binomial", alpha = 0, standardize = FALSE)
  cv_l1 <- cv.glmnet(X, y, family = "binomial", alpha = 1, standardize = FALSE)
  eta_l2 <- as.numeric(predict(cv_l2, newx = X, s = "lambda.min", type = "link"))
  eta_l1 <- as.numeric(predict(cv_l1, newx = X, s = "lambda.min", type = "link"))

  ## Firth logistic regression.
  ## pl = FALSE skips profile-likelihood CIs, which we never use here and
  ## which otherwise emit harmless-but-noisy non-convergence warnings under
  ## strong separation.
  firth_fit <- logistf(yb ~ x, pl = FALSE)
  eta_firth <- coef(firth_fit)[1] + coef(firth_fit)[2] * x

  ## Platt scaling (standard and 2N-target variant). type = "link" again
  ## keeps everything on the log-odds scale.
  Np <- sum(yb == 1); Nn <- sum(yb == 0)
  t_std <- ifelse(yb == 1, (Np + 1) / (Np + 2), 1 / (Nn + 2))
  t_2N  <- ifelse(yb == 1, (2 * Np + 1) / (2 * Np + 2), 1 / (2 * Nn + 2))
  eta_platt  <- predict(suppressWarnings(glm(t_std ~ x, family = binomial)), type = "link")
  eta_platt2 <- predict(suppressWarnings(glm(t_2N ~ x, family = binomial)), type = "link")

  ## LDA. With one predictor, equal class priors, and a pooled within-class
  ## variance, LDA's posterior is exactly plogis(a_lda + b_lda * x) for the
  ## same reason p_true is a sigmoid above -- just with sample mean/variance
  ## in place of the true population parameters. Computing it this way
  ## (rather than via MASS::lda's predict(), which returns linear-scale
  ## posteriors) keeps it on the log scale too.
  m1 <- mean(x1); m2 <- mean(x2)
  s2_pooled <- ((n - 1) * var(x1) + (n - 1) * var(x2)) / (2 * n - 2)
  b_lda <- (m2 - m1) / s2_pooled
  a_lda <- -(m2^2 - m1^2) / (2 * s2_pooled)
  eta_lda <- a_lda + b_lda * x

  ## Bayesian logistic regression, posterior mode via arm::bayesglm, with
  ## Gelman, Jakulin, Pittau & Su's (2008) data-INDEPENDENT weakly
  ## informative default: independent Cauchy(0, 2.5) priors on the
  ## coefficients and Cauchy(0, 10) on the intercept, after bayesglm's
  ## internal rescaling of the predictor (scaled = TRUE, their
  ## recommendation). Unlike Firth's Jeffreys prior, this prior does not
  ## depend on the design matrix. prior.df = 1 is the Cauchy.
  cauchy_fit <- bayesglm(yb ~ x, family = binomial,
                         prior.scale = 2.5, prior.df = 1,
                         prior.scale.for.intercept = 10,
                         prior.df.for.intercept = 1)
  eta_cauchy <- coef(cauchy_fit)[1] + coef(cauchy_fit)[2] * x

  ## Same construction with a Student-t (df = 4) prior: lighter tails than
  ## the Cauchy (df = 1) but still well short of Normal, so it sits between
  ## Cauchy's robustness and L2/Gaussian shrinkage rather than collapsing
  ## toward L2 the way a higher df (close to the top of the commonly
  ## recommended 3-7 range) would.
  t_fit <- bayesglm(yb ~ x, family = binomial,
                    prior.scale = 2.5, prior.df = 4,
                    prior.scale.for.intercept = 10,
                    prior.df.for.intercept = 4)
  eta_t <- coef(t_fit)[1] + coef(t_fit)[2] * x

  etas <- list(L1 = eta_l1, L2 = eta_l2, Firth = eta_firth,
               Platt = eta_platt, Platt_2N = eta_platt2, LDA = eta_lda,
               Cauchy = eta_cauchy, StudentT = eta_t)

  do.call(rbind, lapply(names(etas), function(model_name) {
    log_p_hat <- plogis(etas[[model_name]], log.p = TRUE)
    data.frame(
      rep         = rep_id,
      point_id    = seq_along(x),
      class       = as.character(y),
      x           = x,
      model       = model_name,
      log_p_true  = log_p_true,
      log_p_hat   = log_p_hat,
      log10_ratio = (log_p_hat - log_p_true) / ln10,
      separable   = separable
    )
  }))
}

results <- do.call(rbind, lapply(seq_len(n_reps), fit_one_rep))

out_file <- path.expand("calibration_ratios.csv")
write.csv(results, out_file, row.names = FALSE)
cat(sprintf("Wrote %s with %d rows (%d reps x %d points x %d models)\n",
            out_file, nrow(results), n_reps, 2 * n, length(unique(results$model))))
stopifnot(length(unique(results$model)) == 8)
