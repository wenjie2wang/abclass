library(abclass)

## follow example 1 in Zhang and Liu (2014)
ntrain <- 100 # size of training set
ntest <- 1000 # size of testing set
p <- 100      # number of predictors
k <- 5        # number of categories

n <- ntrain + ntest
train_idx <- seq_len(ntrain)
set.seed(123)
y <- sample(k, size = n, replace = TRUE)       # response
mu <- matrix(rnorm(p * k), nrow = k, ncol = p) # mean vector
## normalize the mean vector so that they are distributed on the unit circle
mu <- mu / apply(mu, 1, function(a) sqrt(sum(a ^ 2)))
x <- t(sapply(y, function(i) rnorm(p, mean = mu[i, ], sd = 0.25)))
train_x <- x[train_idx, ]
test_x <- x[- train_idx, ]
y <- factor(paste0("label_", y))
train_y <- y[train_idx]
test_y <- y[- train_idx]

## logistic deviance loss by default
model1 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  lambda_min_ratio = 1e-3, rel_tol = 1e-3)
pred1 <- predict(model1, test_x)
table(test_y, pred1)
mean(test_y == pred1) # accuracy

## exponential loss approximating AdaBoost
model2 <- abclass(train_x, train_y, nlambda = 10, nfolds = 3,
                  loss = "boost", rel_tol = 1e-3)
pred2 <- predict(model2, test_x)
table(test_y, pred2)
mean(test_y == pred2) # accuracy
