library(abclass)
set.seed(123)

## toy examples for demonstration purpose
## reference: example 1 in Zhang and Liu (2014)
ntrain <- 100 # size of training set
ntest <- 100  # size of testing set
p0 <- 5       # number of actual predictors
p1 <- 5       # number of random predictors
k <- 5        # number of categories

n <- ntrain + ntest; p <- p0 + p1
train_idx <- seq_len(ntrain)
y <- sample(k, size = n, replace = TRUE)         # response
mu <- matrix(rnorm(p0 * k), nrow = k, ncol = p0) # mean vector
## normalize the mean vector so that they are distributed on the unit circle
mu <- mu / apply(mu, 1, function(a) sqrt(sum(a ^ 2)))
x0 <- t(sapply(y, function(i) rnorm(p0, mean = mu[i, ], sd = 0.25)))
x1 <- matrix(rnorm(p1 * n, sd = 0.3), nrow = n, ncol = p1)
x <- cbind(x0, x1)
train_x <- x[train_idx, ]
test_x <- x[- train_idx, ]
y <- factor(paste0("label_", y))
train_y <- y[train_idx]
test_y <- y[- train_idx]

## Regularization through ridge penalty
control1 <- abclass.control(nlambda = 5, lambda_min_ratio = 1e-3,
                            alpha = 1, grouped = FALSE)
model1 <- abclass(train_x, train_y, loss = "logistic",
                  control = control1)
pred1 <- predict(model1, test_x, s = 5)
table(test_y, pred1)
mean(test_y == pred1) # accuracy

## groupwise regularization via group lasso
model2 <- abclass(train_x, train_y, loss = "boost",
                  grouped = TRUE, nlambda = 5)
pred2 <- predict(model2, test_x, s = 5)
table(test_y, pred2)
mean(test_y == pred2) # accuracy
