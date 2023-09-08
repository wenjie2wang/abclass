##
## R package abclass developed by Wenjie Wang <wang@wwenjie.org>
## Copyright (C) 2021-2023 Eli Lilly and Company
##
## This file is part of the R package abclass.
##
## The R package abclass is free software: You can redistribute it and/or
## modify it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or any later
## version (at your option). See the GNU General Public License at
## <https://www.gnu.org/licenses/> for details.
##
## The R package abclass is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
##

.abrank <- function(x, y, qid,
                    weight = NULL,
                    loss = "logistic",
                    control = abrank.control(),
                    ## cv
                    cv_metric = 0L,
                    ## et
                    nstages = 0L)
{
    ## pre-process
    ## TODO add support to sparse matrices
    ## is_x_sparse <- FALSE
    ## if (inherits(x, "sparseMatrix")) {
    ##     is_x_sparse <- TRUE
    ## } else if (! is.matrix(x)) {
    ##     x <- as.matrix(x)
    ## }
    if (! is.matrix(x)) {
        x <- as.matrix(x)
    }
    ## qid must be {0, 1, 2, ...}
    qid <- as.integer(factor(qid)) - 1L
    if (is.null(control$lambda_min_ratio)) {
        control$lambda_min_ratio <- if (nrow(x) < ncol(x)) 1e-4 else 1e-2
    }
    ## determine the loss and penalty function
    loss_id <- match(loss, c("logistic", "boost", "hinge-boost", "lum"))
    ## prepare arguments
    ctrl <- c(
        control,
        list(weight = null2num0(weight),
             cv_metric = as.integer(cv_metric),
             nstages = as.integer(nstages),
             loss_id = loss_id)
    )
    ## arguments
    call_list <- list(
        x = x,
        y = y,
        qid = qid,
        control = ctrl
    )
    ## fun_to_call <- if (is_x_sparse) {
    ##                    rcpp_abrank_fit_sp
    ##                } else {
    ##                    rcpp_abrank_fit
    ##                }
    fun_to_call <- rcpp_abrank_fit
    res <- do.call(fun_to_call, call_list)
    ## post-process
    res$loss <- loss
    res$control <- control
    if (call_list$control$cv_metric == 0L) {
        res$cross_validation <- NULL
    } else if (call_list$control$cv_metric == 1L) {
        ## add mean over queries
        res$cross_validation$cv_recall_mean <- rowMeans(
            res$cross_validation$cv_recall, dims = 2
        )
    } else {
        ## add mean over queries
        res$cross_validation$cv_delta_recall_mean <- colMeans(
            res$cross_validation$cv_delta_recall
        )
    }
    ## update regularization
    return_lambda <-
        if (call_list$control$nstages == 0L) {
            c("alpha", "lambda", "lambda_max")
        } else {
            ## update the selected index to one-based index
            res$et$selected <- res$et$selected + 1L
            "alpha"
        }
    res$regularization <- res$regularization[return_lambda]
    ## return
    res
}
