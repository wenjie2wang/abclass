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
model1 <- et.abclass(train_x, train_y, nstages = 2,
                     lambda_min_ratio = 1e-6, grouped = FALSE,
                     refit = FALSE)
expect_equivalent(dim(coef(model1)), c(p + 1, k - 1))

## with refit being TRUE
model1 <- et.abclass(train_x, train_y, nstages = 2,
                     lambda_min_ratio = 1e-6, grouped = TRUE,
                     refit = TRUE)
expect_equivalent(dim(coef(model1)), c(p + 1, k - 1))
pred1 <- predict(model1, test_x)
expect_true(mean(test_y == pred1) > 0.5)

## with reift as a list
## with cv
model1 <- et.abclass(train_x, train_y, nstages = 2,
                     lambda_min_ratio = 1e-6, grouped = TRUE,
                     refit = list(alpha = 0, nlambda = 10, nfolds = 3))
expect_equivalent(dim(coef(model1)), c(p + 1, k - 1))
pred1 <- predict(model1, test_x)
expect_true(mean(test_y == pred1) > 0.5)

## without cv
model1 <- et.abclass(train_x, train_y, nstages = 2,
                     lambda_min_ratio = 1e-6, grouped = TRUE,
                     refit = list(alpha = 0, nlambda = 10))
expect_equivalent(dim(coef(model1, selection = 10)), c(p + 1, k - 1))
pred1 <- predict(model1, test_x, s = 10)
expect_true(mean(test_y == pred1) > 0.5)

## incorrect length of group weights
expect_error(
    et.abclass(train_x, train_y, group_weight = runif(ncol(train_x) + 1))
)

## with refit and group weights
gw <- runif(ncol(train_x))
model1 <- et.abclass(train_x, train_y, nstages = 2,
                     lambda_min_ratio = 1e-4,
                     control = list(
                         group_weight = gw
                     ))
expect_equal(gw, model1$regularization$group_weight)
expect_equivalent(dim(coef(model1)), c(p + 1, k - 1))
pred1 <- predict(model1, test_x)
expect_true(mean(test_y == pred1) > 0.5)
