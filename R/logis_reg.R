malc_logistic <- function(x, y, lambda, alpha = 1, penalty_factor = NULL,
                          start = NULL, intercept = TRUE, standardize = TRUE,
                          max_iter = 200, rel_tol = 1e-3, pmin = 1e-5,
                          early_stop = FALSE, verbose = FALSE, ...)
{
    Call <- match.call()
    ## pre-process
    if (! is.matrix(x)) {
        x <- as.matrix(x)
    }
    cat_y <- cat2z(y)
    ## model fitting
    fit <- rcpp_logistic_reg(
        x = x,
        y = as.matrix(cat_y$y),
        lambda = lambda,
        alpha = alpha,
        penalty_factor = null2mat0(penalty_factor),
        start = null2mat0(start),
        intercept = intercept,
        standardize = standardize,
        max_iter = max_iter,
        rel_tol = rel_tol,
        pmin = pmin,
        early_stop = early_stop,
        verbose = verbose
    )
    ## post-process
    fit$category <- cat_y
    fit$category$y <- NULL
    fit$call <- Call
    fit$intercept <- intercept
    fit$control <- list(
        standardize <- standardize,
        max_iter = max_iter,
        rel_tol = rel_tol,
        pmin = pmin,
        early_stop = early_stop,
        verbose = verbose
    )
    class(fit) <- "malc_logistic"
    ## return
    fit
}
