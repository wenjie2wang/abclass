##
## R package abclass developed by Wenjie Wang <wang@wwenjie.org>
## Copyright (C) 2021-2024 Eli Lilly and Company
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

##' MOML with ET-Lasso
##'
##' Tune the regularization parameter for MOML by the ET-Lasso method (Yang, et
##' al., 2019).
##'
##' @inheritParams moml
##' @inheritParams et.abclass
##'
##' @references
##'
##' Yang, S., Wen, J., Zhan, X., & Kifer, D. (2019). ET-Lasso: A new efficient
##' tuning of lasso-type regularization for high-dimensional data. In
##' \emph{Proceedings of the 25th ACM SIGKDD International Conference on
##' Knowledge Discovery & Data Mining} (pp. 607--616).
##'
##' @export
et.moml <- function(x,
                    treatment,
                    reward,
                    propensity_score,
                    intercept = TRUE,
                    weight = NULL,
                    loss = c("logistic", "boost", "hinge-boost", "lum"),
                    control = list(),
                    nstages = 2,
                    refit = list(lambda = 1e-6),
                    ...)
{
    ## nstages
    nstages <- as.integer(nstages)
    if (nstages < 1L) {
        stop("The 'nstages' must be a positive integer.")
    }
    ## loss
    all_loss <- c("logistic", "boost", "hinge-boost", "lum")
    loss <- match.arg(loss, choices = all_loss)
    ## controls
    dot_list <- list(...)
    control <- do.call(moml.control, modify_list(control, dot_list))
    ## prepare arguments
    res <- .moml(
        x = x,
        treatment,
        reward,
        propensity_score,
        intercept = intercept,
        loss = loss,
        control = control,
        nstages = nstages
    )
    ## refit if needed
    if (! isFALSE(refit) && length(res$et$selected) > 0) {
        if (isTRUE(refit)) {
            refit <- list(lambda = 1e-6)
        }
        idx <- res$et$selected
        ## inherit the penalty factors for those selected predictors
        if (! is.null(res$regularization$penalty_factor)) {
            refit$penalty_factor <- res$regularization$penalty_factor[idx]
        }
        refit_control <- modify_list(control, refit)
        refit_res <- .moml(
            x = x[, idx, drop = FALSE],
            treatment,
            reward,
            propensity_score,
            ## assume intercept, weight, loss are the same with et-lasso
            intercept = intercept,
            loss = loss,
            control = refit_control,
            ## cv
            nfolds = null0(refit$nfolds),
            stratified = null0(refit$stratified),
            alignment = null0(refit$alignment),
            ## et
            nstages = null0(refit$nstages)
        )
        if (! is.null(refit_res$cross_validation)) {
            ## add cv idx
            cv_idx_list <- with(refit_res$cross_validation,
                                select_lambda(cv_accuracy_mean, cv_accuracy_sd))
            refit_res$cross_validation <- c(refit_res$cross_validation,
                                            cv_idx_list)
        }
        res$refit <- refit_res[! names(refit_res) %in%
                               c("intercept", "loss", "category")]
        res$refit$selected_coef <- idx
    } else {
        res$refit <- FALSE
    }
    ## add class
    class(res) <- c("et.moml", "moml", "abclass")
    ## return
    res
}
