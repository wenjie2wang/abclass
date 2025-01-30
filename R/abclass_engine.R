##
## R package abclass developed by Wenjie Wang <wang@wwenjie.org>
## Copyright (C) 2021-2025 Eli Lilly and Company
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

## encode loss and penalty functions
.all_abclass_losses <- c(
    "logistic", "boost", "hinge.boost", "lum", "mlogit",
    "mle.logistic", "mle.boost", "mle.hinge.boost", "mle.lum"
)
.id_loss <- function(loss)
{
    match(loss, .all_abclass_losses)
}
.all_abclass_penalties <- c("lasso", "scad", "mcp",
                            "glasso", "gscad", "gmcp",
                            "cmcp", "gel", "mlasso", "mmcp")
.id_penalty <- function(penalty)
{
    match(penalty, .all_abclass_penalties)
}

## engine function that should be called internally only
.abclass <- function(x, y,
                     loss,
                     penalty,
                     weights = NULL,
                     offset = NULL,
                     intercept = TRUE,
                     ## abclass.control
                     control = abclass.control(),
                     ## cv
                     nfolds = 0L,
                     stratified = TRUE,
                     alignment = 0L,
                     ## et
                     nstages = 0L,
                     ## moml
                     moml_args = NULL)
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
    ## better to append new options to the existing ones
    loss_id <- .id_loss(loss)
    penalty_id <- .id_penalty(penalty)
    ## process alignment
    all_alignment <- c("fraction", "lambda")
    if (is.numeric(alignment)) {
        alignment <- as.integer(alignment[1L])
    } else if (is.character(alignment)) {
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
        list(loss_id = loss_id,
             penalty_id = penalty_id,
             weights = null2num0(weights),
             offset = null2mat0(offset),
             intercept = intercept,
             nfolds = as.integer(nfolds),
             stratified = stratified,
             alignment = as.integer(alignment),
             nstages = as.integer(nstages),
             owl_reward = null2num0(moml_args$reward))
    )
    ctrl$lambda <- null2num0(ctrl$lambda)
    ctrl$penalty_factor = null2num0(ctrl$penalty_factor)
    ## set up weights in the outcome-weighted learning
    if (length(moml_args) > 0) {
        owl_weight <- abs(moml_args$reward) / moml_args$propensity_score
        if (length(ctrl$weights) > 0) {
            ctrl$weights <- ctrl$weights * owl_weight
        } else {
            ctrl$weights <- owl_weight
        }
    }
    ## main
    call_list <- list(x = x, y = cat_y$y, control = ctrl)
    fun_to_call <- if (is_x_sparse) {
                       rcpp_abclass_fit_sp
                   } else {
                       rcpp_abclass_fit
                   }
    res <- do.call(fun_to_call, call_list)
    ## post-process
    res$category <- cat_y
    res$category$k <- length(res$category$label)
    res$specs <- list(
        loss = loss,
        penalty = penalty,
        weights = res$weights,
        offset = res$offset,
        intercept = intercept
    )
    if (is.null(weights)) {
        res$specs["weights"] <- list(NULL)
    }
    if (is.null(offset)) {
        res$specs["offset"] <- list(NULL)
    }
    res$weights <- NULL
    res$offset <- NULL
    res$control <- control
    if (call_list$control$nfolds == 0L) {
        res$cross_validation <- NULL
    } else {
        res$cross_validation$alignment <- all_alignment[
            res$cross_validation$alignment + 1L
        ]
    }
    if (call_list$control$nstages == 0L) {
        res$et <- NULL
    } else {
        ## update the selected index to one-based index
        res$et$selected <- res$et$selected + 1L
    }
    ## update regularization
    return_lambda <- c("alpha", "lambda", "penalty_factor",
                       "lambda_max", "l1_lambda_max")
    if (grepl("scad|mcp", penalty)) {
        return_lambda <- c(return_lambda, "ncv_kappa", "ncv_gamma")
    } else if (penalty == "gel") {
        return_lambda <- c(return_lambda, "gel_tau")
    } else if (grepl("^mellow", penalty)) {
        return_lambda <- c(return_lambda, "mellowmax_omega")
    }
    res$regularization <- res$regularization[return_lambda]
    ## return
    res
}
