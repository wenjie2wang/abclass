## shared setup
set.seed(123)
ntrain <- 100
ntest  <- 500
p <- 2
k <- 3
n <- ntrain + ntest
train_idx <- seq_len(ntrain)
y_int <- sample(k, size = n, replace = TRUE)
mu    <- matrix(rnorm(p * k), nrow = k, ncol = p)
mu    <- mu / apply(mu, 1, function(a) sqrt(sum(a ^ 2)))
x     <- t(sapply(y_int, function(i) rnorm(p, mean = mu[i, ], sd = 0.25)))
train_x <- x[train_idx, ]
test_x  <- x[- train_idx, ]
y       <- factor(paste0("label_", y_int))
train_y <- y[train_idx]
test_y  <- y[- train_idx]

## -----------------------------------------------------------------------
## Additional penalties
## -----------------------------------------------------------------------
for (pen in c("scad", "gscad", "mcp", "gmcp")) {
    m <- abclass(train_x, train_y, lambda = 0.01, penalty = pen)
    expect_true(inherits(m, "abclass"),
                info = paste("penalty =", pen))
    pred <- predict(m, test_x)
    expect_equal(length(pred), ntest,
                 info = paste("predict length, penalty =", pen))
}

## -----------------------------------------------------------------------
## intercept = FALSE
## -----------------------------------------------------------------------
m_no_int <- abclass(train_x, train_y, nlambda = 5, intercept = FALSE)
## coef excludes the intercept row when intercept = FALSE
expect_equivalent(dim(coef(m_no_int, s = 5)), c(p, k - 1L))
pred_no_int <- predict(m_no_int, test_x, s = 5)
expect_equal(length(pred_no_int), ntest)

## -----------------------------------------------------------------------
## weights parameter
## -----------------------------------------------------------------------
w <- runif(ntrain, 0.5, 1.5)
m_w <- abclass(train_x, train_y, nlambda = 5, weights = w)
expect_true(inherits(m_w, "abclass"))
pred_w <- predict(m_w, test_x, s = 5)
expect_equal(length(pred_w), ntest)

## -----------------------------------------------------------------------
## standardize = FALSE
## -----------------------------------------------------------------------
m_std <- abclass(train_x, train_y, nlambda = 5,
                 control = abclass.control(standardize = FALSE))
expect_true(inherits(m_std, "abclass"))

## -----------------------------------------------------------------------
## LUM control parameters (lum_a, lum_c)
## -----------------------------------------------------------------------
m_lum <- abclass(train_x, train_y, nlambda = 5, loss = "lum",
                 control = abclass.control(lum_a = 2, lum_c = 1))
expect_true(inherits(m_lum, "abclass"))
expect_equivalent(dim(coef(m_lum, s = 5)), c(p + 1L, k - 1L))

## -----------------------------------------------------------------------
## predict "all" selection returns list
## -----------------------------------------------------------------------
m5 <- abclass(train_x, train_y, nlambda = 5, loss = "logistic")
pred_all <- predict(m5, test_x, s = "all")
expect_equal(length(pred_all), 5L)

## -----------------------------------------------------------------------
## cv.abclass: unstratified folds, lambda alignment
## -----------------------------------------------------------------------
cv_unstrat <- cv.abclass(train_x, train_y, nlambda = 5,
                          nfolds = 3, stratified = FALSE)
expect_true(inherits(cv_unstrat, "cv.abclass"))
pred_cv <- predict(cv_unstrat, test_x)
expect_equal(length(pred_cv), ntest)

cv_lalign <- cv.abclass(train_x, train_y, nlambda = 5,
                         nfolds = 3, alignment = "lambda")
expect_true(inherits(cv_lalign, "cv.abclass"))

## -----------------------------------------------------------------------
## BIC.supclass()
## -----------------------------------------------------------------------
sm <- supclass(train_x, train_y, model = "logistic", penalty = "lasso",
               lambda = c(0.05, 0.1, 0.2))
bic_vals <- BIC(sm)
expect_equal(length(bic_vals), 3L)
expect_true(all(is.finite(bic_vals)))

## psvm has no negLogL, BIC should return NULL
sm_psvm <- supclass(train_x, train_y, model = "psvm", penalty = "lasso",
                    lambda = c(0.05, 0.1))
expect_null(BIC(sm_psvm))

## -----------------------------------------------------------------------
## abclass.control() round-trips
## -----------------------------------------------------------------------
ctrl <- abclass.control(alpha = 0.5, nlambda = 10, maxit = 200L,
                        epsilon = 1e-5, standardize = TRUE)
expect_equal(ctrl$alpha, 0.5)
expect_equal(ctrl$nlambda, 10L)
expect_equal(ctrl$maxit, 200L)
