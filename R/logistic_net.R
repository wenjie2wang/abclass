logistic_net <- function(x, y,
                         lambda,
                         alpha = 0.5,
                         start = NULL,
                         weight = NULL,
                         intercept = TRUE,
                         standardize = TRUE,
                         max_iter = 1e4,
                         rel_tol = 1e-5,
                         pmin = 1e-5,
                         verbose = FALSE,
                         ...)
{
    Call <- match.call()
    ## pre-process
    if (! is.matrix(x)) {
        x <- as.matrix(x)
    }
    cat_y <- cat2z(y)
    ## model fitting
    fit <- rcpp_logistic_net(
        x = x,
        y = cat_y$y - 1L,
        lambda = lambda,
        alpha = alpha,
        start = null2mat0(start),
        weight = null2num0(weight),
        intercept = intercept,
        standardize = standardize,
        max_iter = max_iter,
        rel_tol = rel_tol,
        pmin = pmin,
        verbose = verbose
    )
    ## post-process
    fit$category <- cat_y
    fit$call <- Call
    fit$intercept <- intercept
    fit$control <- list(
        standardize <- standardize,
        start = start,
        max_iter = max_iter,
        rel_tol = rel_tol,
        pmin = pmin,
        verbose = verbose
    )
    class(fit) <- "abclass_logistic_net"
    ## return
    fit
}


logistic_net_path <- function(x, y,
                              lambda = NULL,
                              alpha = 0.5,
                              nlambda = 100,
                              lambda_min_ratio = 1e-6,
                              weight = NULL,
                              intercept = TRUE,
                              standardize = TRUE,
                              max_iter = 1e4,
                              rel_tol = 1e-5,
                              pmin = 1e-5,
                              verbose = FALSE,
                              ...)
{
    Call <- match.call()
    ## pre-process
    if (! is.matrix(x)) {
        x <- as.matrix(x)
    }
    cat_y <- cat2z(y)
    if (is.null(lambda_min_ratio)) {
        lambda_min_ratio <- if (nrow(x) < ncol(x)) 1e-4 else 1e-2
    }
    ## model fitting
    fit <- rcpp_logistic_net_path(
        x = x,
        y = cat_y$y - 1L,
        lambda = null2num0(lambda),
        alpha = alpha,
        nlambda = nlambda,
        lambda_min_ratio = lambda_min_ratio,
        weight = null2num0(weight),
        intercept = intercept,
        standardize = standardize,
        max_iter = max_iter,
        rel_tol = rel_tol,
        pmin = pmin,
        verbose = verbose
    )
    ## post-process
    fit$category <- cat_y
    fit$call <- Call
    fit$intercept <- intercept
    fit$control <- list(
        standardize = standardize,
        max_iter = max_iter,
        rel_tol = rel_tol,
        pmin = pmin,
        verbose = verbose
    )
    class(fit) <- "abclass_logistic_net_path"
    ## return
    fit
}
