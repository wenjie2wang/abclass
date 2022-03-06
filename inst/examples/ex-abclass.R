library(abclass)
set.seed(123)

## toy examples for demonstration purpose
## reference: example 1 in Zhang and Liu (2014)
ntrain <- 100 # size of training set
ntest <- 1000 # size of testing set
p0 <- 10      # number of actual predictors
p1 <- 100     # number of random predictors
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

### Regularization through elastic-net penalty
## logistic deviance loss
model1 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  loss = "logistic", lambda_min_ratio = 1e-4)
pred1 <- predict(model1, test_x)
table(test_y, pred1)
mean(test_y == pred1) # accuracy

## exponential loss approximating AdaBoost
model2 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  loss = "boost", rel_tol = 1e-3)
pred2 <- predict(model2, test_x)
table(test_y, pred2)
mean(test_y == pred2) # accuracy

## hybrid hinge-boost loss
model3 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  loss = "hinge-boost", rel_tol = 1e-3)
pred3 <- predict(model3, test_x)
table(test_y, pred3)
mean(test_y == pred3) # accuracy

## large-margin unified loss
model4 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  loss = "lum", rel_tol = 1e-3)
pred4 <- predict(model4, test_x)
table(test_y, pred4)
mean(test_y == pred4) # accuracy

### groupwise regularization via group lasso
## logistic deviance loss
model1 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  grouped = TRUE, loss = "logistic", rel_tol = 1e-3)
pred1 <- predict(model1, test_x)
table(test_y, pred1)
mean(test_y == pred1) # accuracy

## exponential loss approximating AdaBoost
model2 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  grouped = TRUE, loss = "boost", rel_tol = 1e-3)
pred2 <- predict(model2, test_x)
table(test_y, pred2)
mean(test_y == pred2) # accuracy

## hybrid hinge-boost loss
model3 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  grouped = TRUE, loss = "hinge-boost", rel_tol = 1e-3)
pred3 <- predict(model3, test_x)
table(test_y, pred3)
mean(test_y == pred3) # accuracy

## large-margin unified loss
model4 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  grouped = TRUE, loss = "lum", rel_tol = 1e-3)
pred4 <- predict(model4, test_x)
table(test_y, pred4)
mean(test_y == pred4) # accuracy
