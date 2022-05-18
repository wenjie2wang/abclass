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

### regularization through elastic-net penalty
## logistic deviance loss
model1 <- abclass(train_x, train_y, nlambda = 100,
                  nfolds = 3, loss = "logistic")
pred1 <- predict(model1, test_x)
table(test_y, pred1)
```

    ##          pred1
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     179       1      26       0       0
    ##   label_2       1     200       1       1       0
    ##   label_3       4       0     195       0       0
    ##   label_4       0       2       4     183       3
    ##   label_5       1       0       3       2     194

``` r
mean(test_y == pred1) # accuracy
```

    ## [1] 0.951

``` r
## exponential loss approximating AdaBoost
model2 <- abclass(train_x, train_y, nlambda = 100,
                  nfolds = 3, loss = "boost")
pred2 <- predict(model2, test_x, s = "cv_1se")
table(test_y, pred2)
```

    ##          pred2
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     184       0      22       0       0
    ##   label_2       0     202       0       1       0
    ##   label_3      18       1     176       3       1
    ##   label_4       1       5       3     177       6
    ##   label_5       1       0       0       1     198

``` r
mean(test_y == pred2) # accuracy
```

    ## [1] 0.937

``` r
## hybrid hinge-boost loss
model3 <- abclass(train_x, train_y, nlambda = 100,
                  nfolds = 3, loss = "hinge-boost")
pred3 <- predict(model3, test_x)
table(test_y, pred3)
```

    ##          pred3
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     179       1      26       0       0
    ##   label_2       1     201       0       1       0
    ##   label_3       5       0     194       0       0
    ##   label_4       0       2       3     185       2
    ##   label_5       1       0       2       2     195

``` r
mean(test_y == pred3) # accuracy
```

    ## [1] 0.954

``` r
## large-margin unified loss
model4 <- abclass(train_x, train_y, nlambda = 100,
                  nfolds = 3, loss = "lum")
pred4 <- predict(model4, test_x)
table(test_y, pred4)
```

    ##          pred4
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     179       1      26       0       0
    ##   label_2       1     201       0       1       0
    ##   label_3       4       0     194       0       1
    ##   label_4       0       2       3     185       2
    ##   label_5       1       0       1       0     198

``` r
mean(test_y == pred4) # accuracy
```

    ## [1] 0.957

``` r
### variable selection via group lasso
## logistic deviance loss
model1 <- abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                  grouped = TRUE, loss = "logistic")
pred1 <- predict(model1, test_x, s = "cv_1se")
table(test_y, pred1)
```

    ##          pred1
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     173       1      32       0       0
    ##   label_2       2     197       3       1       0
    ##   label_3       1       1     197       0       0
    ##   label_4       0       2       8     180       2
    ##   label_5       2       0       5       3     190

``` r
mean(test_y == pred1) # accuracy
```

    ## [1] 0.937

``` r
## exponential loss approximating AdaBoost
model2 <- abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                  grouped = TRUE, loss = "boost")
pred2 <- predict(model2, test_x, s = "cv_1se")
table(test_y, pred2)
```

    ##          pred2
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     189       0      17       0       0
    ##   label_2       1     202       0       0       0
    ##   label_3      11       0     187       0       1
    ##   label_4       0       1       2     181       8
    ##   label_5       1       0       0       0     199

``` r
mean(test_y == pred2) # accuracy
```

    ## [1] 0.958

``` r
## hybrid hinge-boost loss
model3 <- abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                  grouped = TRUE, loss = "hinge-boost")
pred3 <- predict(model3, test_x)
table(test_y, pred3)
```

    ##          pred3
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     174       1      31       0       0
    ##   label_2       0     202       0       1       0
    ##   label_3      10       0     188       0       1
    ##   label_4       0       3       7     181       1
    ##   label_5       1       0       1       3     195

``` r
mean(test_y == pred3) # accuracy
```

    ## [1] 0.94

``` r
## large-margin unified loss
model4 <- abclass(train_x, train_y, nlambda = 100, nfolds = 3,
                  grouped = TRUE, loss = "lum")
pred4 <- predict(model4, test_x)
table(test_y, pred4)
```

    ##          pred4
    ## test_y    label_1 label_2 label_3 label_4 label_5
    ##   label_1     181       1      24       0       0
    ##   label_2       0     202       0       1       0
    ##   label_3       5       0     193       0       1
    ##   label_4       0       2       5     183       2
    ##   label_5       1       0       1       0     198

``` r
mean(test_y == pred4) # accuracy
```

    ## [1] 0.957

## References

-   Zhang, C., & Liu, Y. (2014). Multicategory Angle-Based Large-Margin
    Classification. *Biometrika*, 101(3), 625–640.
-   Liu, Y., Zhang, H. H., & Wu, Y. (2011). Hard or soft classification?
    large-margin unified machines. *Journal of the American Statistical
    Association*, 106(493), 166–177.

## License

[GNU General Public License](https://www.gnu.org/licenses/) (≥ 3)

Copyright holder: Eli Lilly and Company
