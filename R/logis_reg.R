malc_logistic <- function(x, y, lambda, alpha = 1,
                          start = NULL, weight = NULL,
                          intercept = TRUE, standardize = TRUE,
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
        early_stop = early_stop,
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
        early_stop = early_stop,
        verbose = verbose
    )
    class(fit) <- "malc_logistic"
    ## return
    fit
}


malc_logistic_path <- function(x, y, lambda = NULL, alpha = 1, nlambda = 100,
                               lambda_min_ratio = NULL, weight = NULL,
                               nfolds = 0, stratified = TRUE,
                               intercept = TRUE, standardize = TRUE,
                               max_iter = 200, rel_tol = 1e-3, pmin = 1e-5,
                               early_stop = FALSE, verbose = FALSE, ...)
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
    fit <- rcpp_logistic_path(
        x = x,
        y = cat_y$y - 1L,
        lambda = null2num0(lambda),
        alpha = alpha,
        nlambda = nlambda,
        lambda_min_ratio = lambda_min_ratio,
        weight = null2num0(weight),
        nfolds = nfolds,
        stratified = stratified,
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
    fit$call <- Call
    fit$intercept <- intercept
    fit$control <- list(
        standardize = standardize,
        max_iter = max_iter,
        rel_tol = rel_tol,
        pmin = pmin,
        early_stop = early_stop,
        verbose = verbose
    )
    class(fit) <- "malc_logistic_path"
    ## return
    fit
}
