abrank <- function(x, y, qid,
                   weight = NULL,
                   loss = c("logistic", "boost", "hinge-boost", "lum"),
                   control = list(),
                   ...)
{
    all_loss <- c("logistic", "boost", "hinge-boost", "lum")
    loss <- match.arg(as.character(loss), choices = all_loss)
    ## controls
    dot_list <- list(...)
    control <- do.call(abrank.control, modify_list(control, dot_list))
    res <- .abrank(
        x = x,
        y = y,
        qid = qid,
        weight = null2num0(weight),
        loss = loss,
        control = control
    )
    class(res) <- c("abrank_path", "abrank")
    ## return
    res
}

abrank.control <- function(lambda = NULL,
                           alpha = 1.0,
                           nlambda = 50L,
                           lambda_min_ratio = NULL,
                           offset = NULL,
                           query_weight = FALSE,
                           lambda_weight = FALSE,
                           lum_a = 1.0,
                           lum_c = 1.0,
                           boost_umin = - 5.0,
                           maxit = 1e5L,
                           epsilon = 1e-4,
                           standardize = TRUE,
                           varying_active_set = TRUE,
                           verbose = 0L,
                           ...)
{
    structure(list(
        alpha = alpha,
        lambda = null2num0(lambda),
        nlambda = as.integer(nlambda),
        lambda_min_ratio = lambda_min_ratio,
        offset = null2mat0(offset),
        query_weight = query_weight,
        lambda_weight = lambda_weight,
        standardize = standardize,
        maxit = as.integer(maxit),
        epsilon = epsilon,
        varying_active_set = varying_active_set,
        verbose = as.integer(verbose),
        boost_umin = boost_umin,
        lum_a = lum_a,
        lum_c = lum_c
    ), class = "abrank.control")
}
