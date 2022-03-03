abclass
================

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/abclass)](https://CRAN.R-project.org/package=abclass)
[![Build
Status](https://github.com/wenjie2wang/abclass/workflows/R-CMD-check/badge.svg)](https://github.com/wenjie2wang/abclass/actions)
[![codecov](https://codecov.io/gh/wenjie2wang/abclass/branch/main/graph/badge.svg)](https://app.codecov.io/gh/wenjie2wang/abclass)

The package **abclass** provides implementations of the multi-category
angle-based classifiers (Zhang & Liu, 2014) with the large-margin
unified machines (Liu, et al., 2011) for high-dimensional data.

## Installation

One can install the released version from
[CRAN](https://CRAN.R-project.org/package=abclass).

``` r
install.packages("abclass")
```

Alternatively, the version under development can be installed as
follows:

``` r
if (! require(remotes)) install.packages("remotes")
remotes::install_github("wenjie2wang/abclass", upgrade = "never")
```

## Getting Started

A toy example is as follows:

``` r
library(abclass)
set.seed(123)

## follow example 1 in Zhang and Liu (2014)
ntrain <- 100 # size of training set
ntest <- 1000 # size of testing set
p <- 100      # number of predictors
k <- 5        # number of categories

n <- ntrain + ntest
train_idx <- seq_len(ntrain)
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

## model 1 with logistic deviance loss
model1 <- abclass(train_x, train_y, nfolds = 3)
pred1 <- predict(model1, test_x)
table(test_y, pred1)
```

    ##          pred1
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     195       0       4       5       2
    ##   label_2      10     171      13       3       6
    ##   label_3       5       2     184       5       3
    ##   label_4      20       3       7     159       3
    ##   label_5      27      17      12       8     136

``` r
mean(test_y == pred1) # accuracy
```

    ## [1] 0.845

``` r
## model 2 with exponential loss approximating AdaBoost
model2 <- abclass(train_x, train_y, nlambda = 100, nfolds = 3, loss = "boost")
pred2 <- predict(model2, test_x)
table(test_y, pred2)
```

    ##          pred2
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     197       0       3       4       2
    ##   label_2      23     168       7       3       2
    ##   label_3      21       3     164       6       5
    ##   label_4      28       2       3     155       4
    ##   label_5      34       9       6       8     143

``` r
mean(test_y == pred2) # accuracy
```

    ## [1] 0.827

``` r
## model 3 with hybrid hinge-boost loss
model3 <- abclass(train_x, train_y, nfolds = 3, loss = "hinge-boost",
                  lambda_min_ratio = 1e-3)
pred3 <- predict(model3, test_x)
table(test_y, pred3)
```

    ##          pred3
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     199       1       1       4       1
    ##   label_2       7     185       5       2       4
    ##   label_3       5       3     179       7       5
    ##   label_4       9       2       3     176       2
    ##   label_5      18       7       4       7     164

``` r
mean(test_y == pred3) # accuracy
```

    ## [1] 0.903

``` r
## model 4 with the large-margin unified machines
model4 <- abclass(train_x, train_y, nfolds = 3, loss = "lum",
                  alpha = 0.1, lambda_min_ratio = 1e-3)
pred4 <- predict(model4, test_x)
table(test_y, pred4)
```

    ##          pred4
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     204       0       1       1       0
    ##   label_2       6     192       3       0       2
    ##   label_3       2       0     197       0       0
    ##   label_4       8       1       3     176       4
    ##   label_5      12       4       4       2     178

``` r
mean(test_y == pred4) # accuracy
```

    ## [1] 0.947

## References

Zhang, C., & Liu, Y. (2014). Multicategory Angle-Based Large-Margin
Classification. , 101(3), 625–640.

Liu, Y., Zhang, H. H., & Wu, Y. (2011). Hard or soft classification?
large-margin unified machines. , 106(493), 166–177.

## License

[GNU General Public License](https://www.gnu.org/licenses/) (≥ 3)

Copyright holder: Eli Lilly and Company
