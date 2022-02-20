logistic_net <- function(x, y,
                         lambda = NULL,
                         alpha = 0.5,
                         nlambda = 100,
                         lambda_min_ratio = NULL,
                         weight = NULL,
                         intercept = TRUE,
                         standardize = TRUE,
                         max_iter = 1e4,
                         rel_tol = 1e-4,
                         varying_active_set = TRUE,
                         verbose = 0L,
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
    res <- rcpp_logistic_net(
        x = x,
        y = cat_y$y,
        lambda = null2num0(lambda),
        alpha = alpha,
        nlambda = nlambda,
        lambda_min_ratio = lambda_min_ratio,
        weight = null2num0(weight),
        intercept = intercept,
        standardize = standardize,
        max_iter = max_iter,
        rel_tol = rel_tol,
        varying_active_set = varying_active_set,
        verbose = verbose
    )
    ## post-process
    res$category <- cat_y
    res$call <- Call
    res$intercept <- intercept
    res$control <- list(
        standardize = standardize,
        max_iter = max_iter,
        rel_tol = rel_tol,
        varying_active_set = varying_active_set,
        verbose = verbose
    )
    class(res) <- "abclass_logistic_net"
    ## return
    res
}
