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
packageVersion("abclass")
```

    ## [1] '0.5.0.9040'

``` r
## toy examples for demonstration purpose
## reference: example 1 in Zhang and Liu (2014)
ntrain <- 400  # size of training set
ntest <- 10000 # size of testing set
p0 <- 5        # number of actual predictors
p1 <- 45       # number of random predictors
k <- 5         # number of categories

set.seed(1)
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
model1 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 5,
                     loss = "logistic", grouped = FALSE)
pred1 <- predict(model1, test_x)
table(test_y, pred1)
```

    ##          pred1
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    1368       0       2     633       0
    ##   label_2       3    1828       4       0     133
    ##   label_3       3       3    1747       0     199
    ##   label_4       0       8       0    1898     126
    ##   label_5       0      63      35       1    1946

``` r
mean(test_y == pred1) # accuracy
```

    ## [1] 0.8787

``` r
### with groupwise lasso
model2 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 5,
                     loss = "logistic", grouped = TRUE)
pred2 <- predict(model2, test_x)
table(test_y, pred2)
```

    ##          pred2
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    1991       1       3       4       4
    ##   label_2       0    1775       0       0     193
    ##   label_3       2       2    1404       0     544
    ##   label_4       6      26       0    1972      28
    ##   label_5       0       9       5       0    2031

``` r
mean(test_y == pred2) # accuracy
```

    ## [1] 0.9173

``` r
## tuning by ET-Lasso instead of cross-validation
model3 <- et.abclass(train_x, train_y, nlambda = 100,
                     loss = "logistic", grouped = TRUE)
pred3 <- predict(model3, test_x)
table(test_y, pred3)
```

    ##          pred3
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    1982       1      10       9       1
    ##   label_2       0    1843       1       0     124
    ##   label_3       1       5    1733       0     213
    ##   label_4       2       7       0    2006      17
    ##   label_5       0      16      20       0    2009

``` r
mean(test_y == pred3) # accuracy
```

    ## [1] 0.9573

## References

- Zhang, C., & Liu, Y. (2014). Multicategory Angle-Based Large-Margin
  Classification. *Biometrika*, 101(3), 625–640.
- Liu, Y., Zhang, H. H., & Wu, Y. (2011). Hard or soft classification?
  large-margin unified machines. *Journal of the American Statistical
  Association*, 106(493), 166–177.

## License

[GNU General Public License](https://www.gnu.org/licenses/) (≥ 3)

Copyright holder: Eli Lilly and Company
