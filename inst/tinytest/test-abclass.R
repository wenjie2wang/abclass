ntrain <- 100
ntest <- 1000
p <- 100
k <- 5
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
model1 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  lambda_min_ratio = 1e-3, epsilon = 1e-3)
pred1 <- predict(model1, test_x)
expect_true(mean(test_y == pred1) > 0.5)
expect_equivalent(dim(coef(model1)), c(p + 1, k - 1))

## exponential loss approximating AdaBoost
model2 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  loss = "boost", epsilon = 1e-3)
pred2 <- predict(model2, test_x)
expect_true(mean(test_y == pred2) > 0.5)
expect_equivalent(dim(coef(model2)), c(p + 1, k - 1))

## hinge-boost loss
model3 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  loss = "hinge-boost", epsilon = 1e-3)
pred3 <- predict(model3, test_x)
expect_true(mean(test_y == pred3) > 0.5)
expect_equivalent(dim(coef(model3)), c(p + 1, k - 1))

## LUM loss
model4 <- abclass(train_x, train_y, nlambda = 5, nfolds = 3,
                  loss = "lum", epsilon = 1e-2)
pred4 <- predict(model4, test_x)
expect_true(mean(test_y == pred4) > 0.5)
expect_equivalent(dim(coef(model4)), c(p + 1, k - 1))
