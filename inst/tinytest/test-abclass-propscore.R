## shared setup: three treatments, two real predictors
set.seed(123)
ntrain <- 100
p <- 2
k <- 3
set.seed(2)
trt_int   <- sample(k, size = ntrain, replace = TRUE)
mu        <- matrix(rnorm(p * k), nrow = k, ncol = p)
mu        <- mu / apply(mu, 1, function(a) sqrt(sum(a ^ 2)))
train_x   <- t(sapply(trt_int, function(i) rnorm(p, mean = mu[i, ], sd = 0.25)))
treatment <- factor(paste0("trt_", trt_int))

## -----------------------------------------------------------------------
## abclass_propscore(): default ET tuning
## -----------------------------------------------------------------------
ps_et <- abclass_propscore(train_x, treatment)
expect_equal(length(ps_et), ntrain)
expect_true(all(ps_et > 0 & ps_et < 1))
expect_true(!is.null(attr(ps_et, "model")))

## -----------------------------------------------------------------------
## abclass_propscore(): cv_1se tuning
## -----------------------------------------------------------------------
ps_cv1 <- abclass_propscore(train_x, treatment, tuning = "cv_1se",
                             nlambda = 5)
expect_equal(length(ps_cv1), ntrain)
expect_true(all(ps_cv1 > 0 & ps_cv1 < 1))

## -----------------------------------------------------------------------
## abclass_propscore(): cv_min tuning
## -----------------------------------------------------------------------
ps_cvm <- abclass_propscore(train_x, treatment, tuning = "cv_min",
                             nlambda = 5)
expect_equal(length(ps_cvm), ntrain)
expect_true(all(ps_cvm > 0 & ps_cvm < 1))

## -----------------------------------------------------------------------
## abclass_propscore(): single lambda bypasses tuning
## -----------------------------------------------------------------------
ps_single <- abclass_propscore(train_x, treatment,
                                control = list(lambda = 0.05))
expect_equal(length(ps_single), ntrain)
expect_true(all(ps_single > 0 & ps_single < 1))

## -----------------------------------------------------------------------
## abclass_propscore(): boost loss
## -----------------------------------------------------------------------
ps_boost <- abclass_propscore(train_x, treatment, loss = "boost",
                               nlambda = 5)
expect_equal(length(ps_boost), ntrain)
