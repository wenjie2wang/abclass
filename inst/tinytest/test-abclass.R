ntrain <- 100
ntest <- 1000
p <- 10
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
model1 <- abclass(train_x, train_y, nlambda = 10,
                  lambda_min_ratio = 1e-3, epsilon = 1e-3,
                  grouped = FALSE)
pred1 <- predict(model1, test_x, s = 10)
expect_true(mean(test_y == pred1) > 0.5)
expect_equivalent(dim(coef(model1, s = 10)), c(p + 1, k - 1))

## exponential loss approximating AdaBoost
model2 <- abclass(train_x, train_y, nlambda = 10,
                  loss = "boost", epsilon = 1e-3)
pred2 <- predict(model2, test_x, s = 10)
expect_true(mean(test_y == pred2) > 0.5)
expect_equivalent(dim(coef(model2, s = 10)), c(p + 1, k - 1))

## hinge-boost loss
model3 <- abclass(train_x, train_y, nlambda = 10,
                  loss = "hinge-boost", epsilon = 1e-3)
pred3 <- predict(model3, test_x, s = 10)
expect_true(mean(test_y == pred3) > 0.5)
expect_equivalent(dim(coef(model3, s = 10)), c(p + 1, k - 1))

## LUM loss
model4 <- abclass(train_x, train_y, nlambda = 5,
                  loss = "lum", epsilon = 1e-2,
                  group_penalty = "mcp")
pred4 <- predict(model4, test_x)[[5]]
expect_true(mean(test_y == pred4) > 0.5)

## prob
prob <- predict(model4, test_x, type = "prob", s = 1)
prob4 <- predict(model4, test_x, type = "prob")
expect_equal(prob, prob4[[1]])
expect_equal(length(prob4), 5L)
expect_equal(dim(prob), c(ntest, k))

expect_equivalent(dim(coef(model4, s = 5)), c(p + 1, k - 1))
expect_error(predict(model4), "newx")

## test as.matrix
model4 <- abclass(as.data.frame(train_x),
                  train_y, nlambda = 5,
                  loss = "lum", epsilon = 1e-2,
                  group_penalty = "mcp")
expect_equal(predict(model4, as.data.frame(test_x), s = 5), pred4)

## quick tests for other options
## grouped
qset <- expand.grid(loss = c("logistic", "boost", "hinge-boost", "lum"),
                    gpenalty = c("lasso", "mcp", "scad"),
                    KEEP.OUT.ATTRS = FALSE,
                    stringsAsFactors = FALSE)
for (k in seq_len(nrow(qset))) {
    qmodel <- abclass(train_x, train_y,
                      lambda = 0.01,
                      loss = qset$loss[k],
                      group_penalty = qset$gpenalty[k])
    qpred <- predict(qmodel, test_x)
    qprob <- predict(qmodel, test_x, type = "prob")
}

## ungrouped penalty
qset <- expand.grid(loss = c("logistic", "boost", "hinge-boost", "lum"),
                    alpha = c(0, 1),
                    KEEP.OUT.ATTRS = FALSE,
                    stringsAsFactors = FALSE)
for (k in seq_len(nrow(qset))) {
    qmodel <- abclass(train_x, train_y,
                      lambda = 0.01,
                      loss = qset$loss[k],
                      alpha = qset$alpha[k],
                      grouped = FALSE)
    qpred <- predict(qmodel, test_x)
    qprob <- predict(qmodel, test_x, type = "prob")
}

## test sparse matrices
if (requireNamespace("Matrix", quietly = TRUE)) {
    sp_train_x <- as(train_x, "sparseMatrix")
    sp_test_x <- as(test_x, "sparseMatrix")

    sp_model <- abclass(sp_train_x, train_y, nlambda = 5, loss = "lum",
                        epsilon = 1e-2, group_penalty = "lasso")
    expect_equal(predict(sp_model, test_x, s = 5),
                 predict(sp_model, sp_test_x, s = 5))

    ## quick tests
    ## grouped
    qset <- expand.grid(loss = c("logistic", "boost", "hinge-boost", "lum"),
                        gpenalty = c("lasso", "mcp", "scad"),
                        KEEP.OUT.ATTRS = FALSE,
                        stringsAsFactors = FALSE)
    for (k in seq_len(nrow(qset))) {
        qmodel <- abclass(sp_train_x, train_y,
                          lambda = 0.01,
                          loss = qset$loss[k],
                          group_penalty = qset$gpenalty[k])
        qpred <- predict(qmodel, sp_test_x)
        qprob <- predict(qmodel, sp_test_x, type = "prob")
    }

    ## ungrouped penalty
    qset <- expand.grid(loss = c("logistic", "boost", "hinge-boost", "lum"),
                        alpha = c(0, 1),
                        KEEP.OUT.ATTRS = FALSE,
                        stringsAsFactors = FALSE)
    for (k in seq_len(nrow(qset))) {
        qmodel <- abclass(sp_train_x, train_y,
                          lambda = 0.01,
                          loss = qset$loss[k],
                          alpha = qset$alpha[k],
                          grouped = FALSE)
        qpred <- predict(qmodel, sp_test_x)
        qprob <- predict(qmodel, sp_test_x, type = "prob")
    }
}
