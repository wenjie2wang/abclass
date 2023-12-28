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

## engine function that should be called internally only
.abclass <- function(x, y,
                     intercept = TRUE,
                     weight = NULL,
                     loss = "logistic",
                     ## abclass.control
                     control = abclass.control(),
                     ## cv
                     nfolds = 0L,
                     stratified = TRUE,
                     alignment = 0L,
                     ## et
                     nstages = 0L)
{
    ## pre-process
    is_x_sparse <- FALSE
    if (inherits(x, "sparseMatrix")) {
        is_x_sparse <- TRUE
    } else if (! is.matrix(x)) {
        x <- as.matrix(x)
    }
    cat_y <- cat2z(y)
    if (is.null(control$lambda_min_ratio)) {
        control$lambda_min_ratio <- if (nrow(x) > ncol(x)) 1e-4 else 1e-2
    }
    ## determine the loss and penalty function
    loss_id <- match(loss, c("logistic", "boost", "hinge-boost", "lum"))
    penalty_id <- 1
    if (control$grouped) {
        penalty_id <- 1 + match(control$group_penalty,
                                c("lasso", "scad", "mcp"))
    }
    ## process alignment
    if (is.numeric(alignment)) {
        alignment <- as.integer(alignment[1L])
    } else if (is.character(alignment)) {
        all_alignment <- c("fraction", "lambda")
        alignment <- match.arg(alignment, choices = all_alignment)
        alignment <- match(alignment, all_alignment) - 1L
    } else {
        stop("Invalid 'alignment'.")
    }
    ## adjust lambda alignment
    if (alignment == 0L && length(control$lambda) > 0) {
        if (control$verbose) {
            message("Changed to `alignment` = 'lambda' ",
                    "for the specified lambda sequence.")
        }
        alignment <- 1L
    }
    ## prepare arguments
    ctrl <- c(
        control,
        list(intercept = intercept,
             weight = null2num0(weight),
             nfolds = as.integer(nfolds),
             stratified = stratified,
             alignment = as.integer(alignment),
             nstages = as.integer(nstages),
             loss_id = loss_id,
             penalty_id = penalty_id)
    )
    ## arguments
    call_list <- c(list(x = x, y = cat_y$y, control = ctrl))
    fun_to_call <- if (is_x_sparse) {
                       rcpp_abclass_fit_sp
                   } else {
                       rcpp_abclass_fit
                   }
    res <- do.call(fun_to_call, call_list)
    ## post-process
    res$category <- cat_y
    res$category$k <- length(res$category$label)
    res$intercept <- intercept
    res$loss <- loss
    res$control <- control
    if (call_list$control$nfolds == 0L) {
        res$cross_validation <- NULL
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
    res$regularization <-
        if (control$grouped) {
            common_pars <- c(return_lambda, "group_weight")
            if (control$group_penalty == "lasso") {
                res$regularization[common_pars]
            } else {
                res$regularization[c(common_pars, "kappa_ratio", "gamma")]
            }
        } else {
            res$regularization[return_lambda]
        }
    ## return
    res
}
