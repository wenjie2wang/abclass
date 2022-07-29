ntrain <- 50
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

## psvm + suplasso
model <- cv.supclass(train_x,
                     train_y,
                     model = "psvm",
                     penalty = "lasso",
                     nfolds = 3,
                     lambda = c(0.01, 0.02))
pred <- predict(model, test_x)
expect_true(mean(test_y == pred) > 0.5)
expect_equivalent(dim(coef(model)), c(p + 1, k))
