abclass
================

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/abclass)](https://CRAN.R-project.org/package=abclass)
[![Build
Status](https://github.com/wenjie2wang/abclass/workflows/R-CMD-check/badge.svg)](https://github.com/wenjie2wang/abclass/actions)
[![codecov](https://codecov.io/gh/wenjie2wang/abclass/branch/main/graph/badge.svg)](https://app.codecov.io/gh/wenjie2wang/abclass)

The package **abclass** provides implementations of the multi-category
angle-based classifiers (Zhang & Liu, 2014) with the large-margin
unified machines (Liu, et al., 2011) for high-dimensional data.

Notice that the package is still experimental and under active
development.

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

### regularization through elastic-net penalty
## logistic deviance loss
model1 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = FALSE, loss = "logistic")
pred1 <- predict(model1, test_x)
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
## exponential loss approximating AdaBoost
model2 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = FALSE, loss = "boost")
pred2 <- predict(model2, test_x, s = "cv_1se")
table(test_y, pred2)
```

    ##          pred2
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    1885      21       0     142      11
    ##   label_2       7    1773       0      13     262
    ##   label_3     112       8    1876       0       1
    ##   label_4     107      22       5    1672      89
    ##   label_5      16       5       2      16    1955

``` r
mean(test_y == pred2) # accuracy
```

    ## [1] 0.9161

``` r
## hybrid hinge-boost loss
model3 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = FALSE, loss = "hinge-boost")
pred3 <- predict(model3, test_x)
table(test_y, pred3)
```

    ##          pred3
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    1912      18       0     116      13
    ##   label_2       7    1732       1       0     315
    ##   label_3     233      23    1731       0      10
    ##   label_4     105      12       5    1639     134
    ##   label_5      28      41       2       3    1920

``` r
mean(test_y == pred3) # accuracy
```

    ## [1] 0.8934

``` r
## large-margin unified loss
model4 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = FALSE, loss = "lum")
pred4 <- predict(model4, test_x)
table(test_y, pred4)
```

    ##          pred4
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    1921      17       0     108      13
    ##   label_2       7    1721       1       0     326
    ##   label_3     191      18    1778       0      10
    ##   label_4     116      11       6    1618     144
    ##   label_5      19      17       2       6    1950

``` r
mean(test_y == pred4) # accuracy
```

    ## [1] 0.8988

``` r
### variable selection via group lasso
## logistic deviance loss
model1 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = TRUE, loss = "logistic")
pred1 <- predict(model1, test_x, s = "cv_1se")
table(test_y, pred1)
```

    ##          pred1
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    2045       4       0       5       5
    ##   label_2      36    1864       0       1     154
    ##   label_3      67       0    1906      12      12
    ##   label_4     428       5       1    1426      35
    ##   label_5      14      10       1       1    1968

``` r
mean(test_y == pred1) # accuracy
```

    ## [1] 0.9209

``` r
## exponential loss approximating AdaBoost
model2 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = TRUE, loss = "boost")
pred2 <- predict(model2, test_x, s = "cv_1se")
table(test_y, pred2)
```

    ##          pred2
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    2043       6       1       8       1
    ##   label_2      19    1974       0       1      61
    ##   label_3      20       1    1970       4       2
    ##   label_4     350       3       2    1518      22
    ##   label_5       9      15       0       1    1969

``` r
mean(test_y == pred2) # accuracy
```

    ## [1] 0.9474

``` r
## hybrid hinge-boost loss
model3 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = TRUE, loss = "hinge-boost")
pred3 <- predict(model3, test_x)
table(test_y, pred3)
```

    ##          pred3
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    2041       5       1       9       3
    ##   label_2      21    1950       0       1      83
    ##   label_3      32       1    1949       8       7
    ##   label_4     254       1       2    1592      46
    ##   label_5       5       9       0       1    1979

``` r
mean(test_y == pred3) # accuracy
```

    ## [1] 0.9511

``` r
## large-margin unified loss
model4 <- cv.abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                     grouped = TRUE, loss = "lum", alpha = 0.5)
pred4 <- predict(model4, test_x)
table(test_y, pred4)
```

    ##          pred4
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1    2034       7       1       8       9
    ##   label_2      11    1966       0       1      77
    ##   label_3      10       4    1971       7       5
    ##   label_4     209       6       3    1628      49
    ##   label_5       3       9       0       1    1981

``` r
mean(test_y == pred4) # accuracy
```

    ## [1] 0.958

## References

-   Zhang, C., & Liu, Y. (2014). Multicategory Angle-Based Large-Margin
    Classification. *Biometrika*, 101(3), 625–640.
-   Liu, Y., Zhang, H. H., & Wu, Y. (2011). Hard or soft classification?
    large-margin unified machines. *Journal of the American Statistical
    Association*, 106(493), 166–177.

## License

[GNU General Public License](https://www.gnu.org/licenses/) (≥ 3)

Copyright holder: Eli Lilly and Company
