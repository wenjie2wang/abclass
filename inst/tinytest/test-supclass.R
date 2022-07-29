ntrain <- 100
ntest <- 1000
p <- 3
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

## logistic + suplasso
model <- supclass(train_x, train_y, model = "logistic", penalty = "lasso")
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))

## logistic + supscad
model <- supclass(train_x, train_y, model = "logistic", penalty = "scad",
                  maxit = 50, epsilon = 1e-3, scad_a = 10)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))

## psvm + suplasso
model <- supclass(train_x, train_y, model = "psvm", penalty = "lasso",
                  maxit = 50, epsilon = 1e-3)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))

## psvm + supscad
model <- supclass(train_x, train_y, model = "psvm", penalty = "scad",
                  maxit = 50, epsilon = 1e-3)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))

## svm + suplasso
model <- supclass(train_x, train_y, model = "svm", penalty = "lasso",
                  maxit = 50, epsilon = 1e-3)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))

## svm + supscad
model <- supclass(train_x, train_y, model = "svm", penalty = "scad",
                  maxit = 50, epsilon = 1e-3)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))
