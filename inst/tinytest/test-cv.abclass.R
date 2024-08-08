ntrain <- 100
ntest <- 1000
p <- 2
k <- 3
n <- ntrain + ntest
train_idx <- seq_len(ntrain)
y <- sample(k, size = n, replace = TRUE)
mu <- matrix(rnorm(p * k), nrow = k, ncol = p)
## normalize the mean vector so that they are distributed on the unit circle
mu <- mu / apply(mu, 1, function(a) sqrt(sum(a ^ 2)))
x <- t(sapply(y, function(i) rnorm(p, mean = mu[i, ], sd = 0.25)))
train_x <- x[train_idx, ]
test_x <- x[- train_idx, ]
y <- factor(paste0("label_", y))
train_y <- y[train_idx]
test_y <- y[- train_idx]

## logistic deviance loss
model1 <- cv.abclass(train_x, train_y, nlambda = 5,
                     lambda_min_ratio = 1e-3, nfolds = 3)
pred1 <- predict(model1, test_x)
expect_true(mean(test_y == pred1) > 0.5)
expect_equivalent(dim(coef(model1, s = "cv_1se")), c(p + 1, k - 1))

## exponential loss approximating AdaBoost
model2 <- cv.abclass(train_x, train_y, nlambda = 5, loss = "boost")
pred2 <- predict(model2, test_x)
expect_true(mean(test_y == pred2) > 0.5)
expect_equivalent(dim(coef(model2, s = 2)), c(p + 1, k - 1))

## hinge.boost loss
model3 <- cv.abclass(train_x, train_y, nlambda = 5,
                     loss = "hinge.boost")
pred3 <- predict(model3, test_x)
expect_true(mean(test_y == pred3) > 0.5)
expect_equivalent(dim(coef(model3, s = 3)), c(p + 1, k - 1))

## LUM loss
model4 <- cv.abclass(train_x, train_y, nlambda = 5,
                     loss = "lum", penalty = "glasso")
pred4 <- predict(model4, test_x, s = "cv_1se")
expect_true(mean(test_y == pred4) > 0.5)
expect_equivalent(dim(coef(model4, s = 5)), c(p + 1, k - 1))

## default refit
model <- cv.abclass(train_x, train_y, nlambda = 5, refit = TRUE)
expect_equivalent(dim(coef(model)), c(p + 1, k - 1))
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)

## refit with cross-validation
model <- cv.abclass(train_x, train_y,
                    nlambda = 5,
                    refit = list(nfolds = 5,
                                 nlambda = 5,
                                 lambda_min_ratio = 1e-4,
                                 alpha = 0)
                    )

## cv_1se (by default)
expect_equivalent(dim(coef(model)), c(p + 1, k - 1))
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)

## cv_min
expect_equivalent(dim(coef(model, s = "cv_min")), c(p + 1, k - 1))
pred <- predict(model, test_x, s = "cv_min")
expect_true(mean(test_y == pred) > 0.5)

## all
expect_equivalent(dim(coef(model, s = "all")), c(p + 1, k - 1, 5))
