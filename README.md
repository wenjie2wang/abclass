abclass
================

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/abclass)](https://CRAN.R-project.org/package=abclass)
[![Build
Status](https://github.com/wenjie2wang/abclass/workflows/R-CMD-check/badge.svg)](https://github.com/wenjie2wang/abclass/actions)
[![codecov](https://codecov.io/gh/wenjie2wang/abclass/branch/main/graph/badge.svg)](https://app.codecov.io/gh/wenjie2wang/abclass)

The package **abclass** provides implementations of the multi-category
angle-based classifiers (Zhang & Liu, 2014) with the large-margin
unified machines (Liu, et al., 2011) for high-dimensional data.

> **Note** This package is still very experimental and under active
> development. The function interface is subject to change without
> guarantee of backward compatibility.

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

## toy examples for demonstration purpose
## reference: example 1 in Zhang and Liu (2014)
ntrain <- 200  # size of training set
ntest <- 10000 # size of testing set
p0 <- 10       # number of actual predictors
p1 <- 100      # number of random predictors
k <- 5         # number of categories

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

### logistic deviance loss with elastic-net penalty
model1 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = FALSE, loss = "logistic")
pred1 <- predict(model1, test_x, s = "cv_min")
table(test_y, pred1)
```

    ##          pred1
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    1879      20       0     143      17
    ##   label_2       8    1638       0       0     409
    ##   label_3     308      23    1652       0      14
    ##   label_4     111      10       5    1617     152
    ##   label_5      33      29       3       5    1924

``` r
mean(test_y == pred1) # accuracy
```

    ## [1] 0.871

``` r
### hinge-boost loss with groupwise lasso
model2 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = TRUE, loss = "hinge-boost")
pred2 <- predict(model2, test_x, s = "cv_1se")
table(test_y, pred2)
```

    ##          pred2
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    2046       4       0       4       5
    ##   label_2      43    1826       0       1     185
    ##   label_3      83       0    1887      13      14
    ##   label_4     476       6       1    1381      31
    ##   label_5      19      12       3       0    1960

``` r
mean(test_y == pred2) # accuracy
```

    ## [1] 0.91

``` r
## tuning by ET-Lasso instead of cross-validation
model3 <- et.abclass(train_x, train_y, nlambda = 100,
                     loss = "lum", alpha = 0.5)
pred3 <- predict(model3, test_x)
table(test_y, pred3)
```

    ##          pred3
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    2033       8       4       6       8
    ##   label_2       7    1993       1       1      53
    ##   label_3       4       2    1989       0       2
    ##   label_4     194      22      12    1622      45
    ##   label_5       6      15       0       2    1971

``` r
mean(test_y == pred3) # accuracy
```

    ## [1] 0.9608

## References

-   Zhang, C., & Liu, Y. (2014). Multicategory Angle-Based Large-Margin
    Classification. *Biometrika*, 101(3), 625–640.
-   Liu, Y., Zhang, H. H., & Wu, Y. (2011). Hard or soft classification?
    large-margin unified machines. *Journal of the American Statistical
    Association*, 106(493), 166–177.

## License

[GNU General Public License](https://www.gnu.org/licenses/) (≥ 3)

Copyright holder: Eli Lilly and Company
