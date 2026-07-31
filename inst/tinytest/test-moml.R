## shared setup: three treatments, two real predictors
ntrain <- 100
ntest  <- 500
p <- 2
k <- 3
n <- ntrain + ntest
train_idx <- seq_len(ntrain)
set.seed(1)
trt_int <- sample(k, size = n, replace = TRUE)
mu <- matrix(rnorm(p * k), nrow = k, ncol = p)
mu <- mu / apply(mu, 1, function(a) sqrt(sum(a ^ 2)))
x  <- t(sapply(trt_int, function(i) rnorm(p, mean = mu[i, ], sd = 0.25)))
train_x   <- x[train_idx, ]
test_x    <- x[- train_idx, ]
treatment <- factor(paste0("trt_", trt_int))
train_trt <- treatment[train_idx]
## rewards and propensity scores (uniform)
reward    <- rnorm(ntrain)
ps        <- rep(1 / k, ntrain)

## -----------------------------------------------------------------------
## moml(): basic fit
## -----------------------------------------------------------------------
m1 <- moml(train_x, train_trt, reward = reward,
           propensity_score = ps, nlambda = 5)
expect_true(inherits(m1, "moml"))
expect_true(inherits(m1, "abclass_path"))
## coefficient dimensions: (p+1) rows x (k-1) cols x nlambda slices
expect_equivalent(dim(coef(m1, s = 5)), c(p + 1L, k - 1L))
pred1 <- predict(m1, test_x, s = 5)
expect_equal(length(pred1), ntest)
expect_true(all(pred1 %in% levels(train_trt)))

## moml(): prediction types
prob1 <- predict(m1, test_x, type = "prob", s = 5)
expect_equal(dim(prob1), c(ntest, k))
link1 <- predict(m1, test_x, type = "link", s = 5)
## link: ntest rows, (k-1) cols
expect_equal(nrow(link1), ntest)

## moml(): boost loss
m2 <- moml(train_x, train_trt, reward = reward,
           propensity_score = ps, nlambda = 5, loss = "boost")
expect_true(inherits(m2, "moml"))

## moml(): lasso penalty
m3 <- moml(train_x, train_trt, reward = reward,
           propensity_score = ps, nlambda = 5, penalty = "lasso")
expect_true(inherits(m3, "moml"))
expect_equivalent(dim(coef(m3, s = 3)), c(p + 1L, k - 1L))

## -----------------------------------------------------------------------
## cv.moml(): cross-validated tuning
## -----------------------------------------------------------------------
cm1 <- cv.moml(train_x, train_trt, reward = reward,
               propensity_score = ps, nlambda = 5, nfolds = 3)
expect_true(inherits(cm1, "cv.moml"))
expect_true(inherits(cm1, "moml"))
pred_cm1 <- predict(cm1, test_x)
expect_equal(length(pred_cm1), ntest)
expect_equivalent(dim(coef(cm1, s = "cv_1se")), c(p + 1L, k - 1L))
expect_equivalent(dim(coef(cm1, s = "cv_min")), c(p + 1L, k - 1L))
expect_equivalent(dim(coef(cm1, s = "all")),    c(p + 1L, k - 1L, 5L))

## cv.moml(): refit = TRUE
cm2 <- cv.moml(train_x, train_trt, reward = reward,
               propensity_score = ps, nlambda = 5,
               nfolds = 3, refit = TRUE)
expect_true(inherits(cm2, "cv.moml"))
expect_equivalent(dim(coef(cm2)), c(p + 1L, k - 1L))

## -----------------------------------------------------------------------
## et.moml(): ET-Lasso tuning
## -----------------------------------------------------------------------
em1 <- et.moml(train_x, train_trt, reward = reward,
               propensity_score = ps, refit = FALSE)
expect_true(inherits(em1, "et.moml"))
expect_true(inherits(em1, "moml"))
expect_equivalent(dim(coef(em1)), c(p + 1L, k - 1L))

## et.moml(): refit = TRUE
em2 <- et.moml(train_x, train_trt, reward = reward,
               propensity_score = ps, refit = TRUE)
expect_true(inherits(em2, "et.moml"))
pred_em2 <- predict(em2, test_x)
expect_equal(length(pred_em2), ntest)

## et.moml(): hinge.boost loss, lasso penalty
em3 <- et.moml(train_x, train_trt, reward = reward,
               propensity_score = ps,
               loss = "hinge.boost", penalty = "lasso",
               refit = FALSE)
expect_true(inherits(em3, "et.moml"))
