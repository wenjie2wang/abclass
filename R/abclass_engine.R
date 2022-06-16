## engine function that should be called internally only
.abclass <- function(x, y,
                     intercept = TRUE,
                     weight = NULL,
                     loss = "logistic",
                     ## regularization
                     lambda = NULL,
                     alpha = 0.5,
                     nlambda = 50L,
                     lambda_min_ratio = NULL,
                     grouped = TRUE,
                     group_weight = NULL,
                     group_penalty = "lasso",
                     dgamma = 1.0,
                     ## loss
                     lum_a = 1.0,
                     lum_c = 1.0,
                     boost_umin = - 5.0,
                     ## control
                     maxit = 1e5L,
                     epsilon = 1e-3,
                     standardize = TRUE,
                     varying_active_set = TRUE,
                     verbose = 0,
                     ## cv
                     nfolds = 0L,
                     stratified = TRUE,
                     alignment = 0L,
                     ## et
                     nstages = 0L,
                     ## internal
                     main_fit = TRUE)
{
    ## pre-process
    is_x_sparse <- FALSE
    if (inherits(x, "sparseMatrix")) {
        is_x_sparse <- TRUE
    } else if (! is.matrix(x)) {
        x <- as.matrix(x)
    }
    cat_y <- cat2z(y)
    if (is.null(lambda_min_ratio)) {
        lambda_min_ratio <- if (nrow(x) < ncol(x)) 1e-4 else 1e-2
    }
    ## prepare arguments
    default_args_to_call <- list(
        x = x,
        y = cat_y$y,
        intercept = intercept,
        weight = null2num0(weight),
        lambda = null2num0(lambda),
        alpha = alpha,
        nlambda = as.integer(nlambda),
        lambda_min_ratio = lambda_min_ratio,
        group_weight = null2num0(group_weight),
        dgamma = dgamma,
        lum_a = lum_a,
        lum_c = lum_c,
        boost_umin = boost_umin,
        maxit = as.integer(maxit),
        epsilon = epsilon,
        standardize = standardize,
        varying_active_set = varying_active_set,
        verbose = as.integer(verbose),
        nfolds = as.integer(nfolds),
        stratified = stratified,
        alignment = as.integer(alignment),
        nstages = as.integer(nstages),
        main_fit = main_fit
    )
    fun_to_call <- if (grouped) {
                       sprintf("r_%s_g%s", loss, group_penalty)
                   } else {
                       sprintf("r_%s_net", loss)
                   }
    if (is_x_sparse) {
        fun_to_call <- paste0(fun_to_call, "_sp")
    }
    args_to_call <- default_args_to_call[
        names(default_args_to_call) %in% formal_names(fun_to_call)
    ]
    res <- do.call(fun_to_call, args_to_call)
    ## post-process
    res$category <- cat_y
    loss2 <- gsub("_", "-", loss, fixed = TRUE)
    res$loss <- switch(
        loss2,
        "logistic" = list(loss = loss2),
        "boost" = list(loss = loss2, boost_umin = boost_umin),
        "hinge-boost" = list(loss = loss2, lum_c = lum_c),
        "lum" = list(loss = loss2, lum_a = lum_a, lum_c = lum_c)
    )
    res$control <- list(
        standardize = standardize,
        maxit = maxit,
        epsilon = epsilon,
        varying_active_set = varying_active_set,
        verbose = verbose
    )
    if (default_args_to_call$nfolds == 0L) {
        res$cross_validation <- NULL
    }
    ## update regularization
    return_lambda <-
        if (default_args_to_call$nstages == 0L) {
            c("lambda", "lambda_max")
        } else {
            ## update the selected index to one-based index
            res$et$selected <- res$et$selected + 1L
            NULL
        }
    res$regularization <-
        if (grouped) {
            common_pars <- c(return_lambda, "group_weight")
            if (group_penalty == "lasso") {
                res$regularization[common_pars]
            } else {
                res$regularization[c(common_pars, "dgamma", "gamma")]
            }
        } else {
            res$regularization[c(return_lambda, "alpha")]
        }
    ## return
    res
}
