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
model <- supclass(train_x, train_y, model = "logistic", penalty = "lasso",
                  lambda = c(0.1, 0.2), epsilon = 1e-3, maxit = 10)
pred <- predict(model, test_x, s = 1)
expect_true(mean(test_y == pred) > 0.5)

## test as.matrix
pred2 <- predict(model, as.data.frame(test_x), s = 1)
expect_equal(pred, pred2)

prob <- predict(model, test_x, type = "prob")
expect_equal(dim(prob), c(ntest, k))
expect_equivalent(dim(coef(model, s = 1)), c(p + 1, k))
expect_error(predict(model), "newx")

## logistic + supscad
model <- supclass(train_x, train_y, model = "logistic", penalty = "scad",
                  maxit = 25, epsilon = 1e-3, scad_a = 10)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))

## psvm + suplasso
model <- supclass(train_x, train_y, model = "psvm", penalty = "lasso",
                  maxit = 25, epsilon = 1e-3)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))

## psvm + supscad
model <- supclass(train_x, train_y, model = "psvm", penalty = "scad",
                  maxit = 25, epsilon = 1e-3)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))

## svm + suplasso
model <- supclass(train_x, train_y, model = "svm", penalty = "lasso",
                  maxit = 25, epsilon = 1e-3)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))

## svm + supscad
model <- supclass(train_x, train_y, model = "svm", penalty = "scad",
                  maxit = 25, epsilon = 1e-3)
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))
