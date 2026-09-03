ntrain <- 100 # size of training set
ntest <- 1000 # size of testing set
p0 <- 2       # number of actual predictors
p1 <- 2       # number of random predictors
k <- 3        # number of categories

n <- ntrain + ntest; p <- p0 + p1
train_idx <- seq_len(ntrain)
y <- sample(k, size = n, replace = TRUE)         # response
mu <- matrix(rnorm(p0 * k), nrow = k, ncol = p0) # mean vector
## normalize the mean vector so that they are distributed on the unit circle
mu <- mu / apply(mu, 1, function(a) sqrt(sum(a ^ 2)))
x0 <- t(sapply(y, function(i) rnorm(p0, mean = mu[i, ], sd = 0.3)))
x1 <- matrix(rnorm(p1 * n, sd = 0.3), nrow = n, ncol = p1)
x <- cbind(x0, x1)
train_x <- x[train_idx, ]
test_x <- x[- train_idx, ]
y <- factor(paste0("label_", y))
train_y <- y[train_idx]
test_y <- y[- train_idx]

## without refit
model1 <- et.abclass(train_x, train_y, refit = FALSE)
expect_equivalent(dim(coef(model1)), c(p + 1, k - 1))

## with refit being TRUE
model1 <- et.abclass(train_x, train_y, refit = TRUE)
expect_equivalent(dim(coef(model1)), c(p + 1, k - 1))
pred1 <- predict(model1, test_x)
expect_true(mean(test_y == pred1) > 0.3)

## with reift as a list
## with cv
model1 <- et.abclass(train_x, train_y,
                     refit = list(alpha = 0, nlambda = 3, nfolds = 3))
expect_equivalent(dim(coef(model1)), c(p + 1, k - 1))
pred1 <- predict(model1, test_x)
expect_true(mean(test_y == pred1) > 0.3)

## without cv
model1 <- et.abclass(train_x, train_y,
                     refit = list(alpha = 0, nlambda = 3))
expect_equivalent(dim(coef(model1, selection = 3)), c(p + 1, k - 1))
pred1 <- predict(model1, test_x, selection = 3)
expect_true(mean(test_y == pred1) > 0.3)

## incorrect length of penalty factors
expect_error(
    et.abclass(train_x, train_y,
               penalty_factor = runif(ncol(train_x) + 1))
)

## with penalty factors
gw <- runif(ncol(train_x))
model1 <- et.abclass(train_x, train_y,
                     control = list(
                         penalty_factor = gw
                     ))
expect_equal(gw, model1$regularization$penalty_factor)
expect_equivalent(dim(coef(model1)), c(p + 1, k - 1))
pred1 <- predict(model1, test_x)
expect_true(mean(test_y == pred1) > 0.3)

## invalid 'relax'
expect_error(et.abclass(train_x, train_y, relax = "bad"))

## relax = TRUE, no cv: coef()/predict() default to relax_gamma = 0
model2 <- et.abclass(train_x, train_y, relax = TRUE)
expect_equal(model2$et$relax_gamma, seq(0, 1, length.out = 11))
manual_beta <- 0 * model2$coefficients[, , 1L, drop = TRUE] +
    1 * model2$relax_coefficients[, , 1L, drop = TRUE]
expect_equal(coef(model2), manual_beta)
expect_equivalent(dim(coef(model2)), c(p + 1, k - 1))
pred2 <- predict(model2, test_x)
expect_true(mean(test_y == pred2) > 0.3)

## coef(..., relax_gamma = "cv_min") errors without cv results
expect_error(coef(model2, relax_gamma = "cv_min"))

## relax = TRUE with cv: relax_gamma selection by "cv_min"/"cv_1se"
model3 <- et.abclass(train_x, train_y, relax = TRUE, nfolds = 3)
expect_equal(length(model3$cross_validation$cv_accuracy_mean),
             length(model3$et$relax_gamma))
coef_min <- coef(model3, relax_gamma = "cv_min")
coef_1se <- coef(model3, relax_gamma = "cv_1se")
expect_equivalent(dim(coef_min), c(p + 1, k - 1))
expect_equivalent(dim(coef_1se), c(p + 1, k - 1))
pred3 <- predict(model3, test_x, relax_gamma = "cv_min")
expect_true(mean(test_y == pred3) > 0.3)

## relax as a list: custom gamma/lambda grid
model4 <- et.abclass(train_x, train_y,
                     relax = list(gamma = c(0, 1), lambda = 1e-3))
expect_equal(model4$et$relax_gamma, c(0, 1))
coef_explicit <- coef(model4, relax_gamma = 0.5)
manual_explicit <- 0.5 * model4$coefficients[, , 1L, drop = TRUE] +
    0.5 * model4$relax_coefficients[, , 1L, drop = TRUE]
expect_equal(coef_explicit, manual_explicit)

