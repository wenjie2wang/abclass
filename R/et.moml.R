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
                    loss = c("logistic", "boost", "hinge.boost", "lum"),
                    penalty = c("glasso", "lasso"),
                    weights = NULL,
                    offset = NULL,
                    intercept = TRUE,
                    control = list(),
                    nstages = 2,
                    nfolds = 0L,
                    stratified = TRUE,
                    alignment = c("fraction", "lambda"),
                    refit = FALSE,
                    ...)
{
    ## nstages
    nstages <- as.integer(nstages)
    if (nstages < 1L) {
        stop("The 'nstages' must be a positive integer.")
    }
    ## loss
    loss <- match.arg(as.character(loss)[1],
                      choices = .all_abclass_losses)
    penalty <- match.arg(as.character(penalty)[1],
                         choices = .all_abclass_penalties)
    ## controls
    dot_list <- list(...)
    control <- do.call(moml.control, modify_list(control, dot_list))
    ## prepare arguments
    res <- .abclass(
        x = x,
        y = treatment,
        loss = loss,
        penalty = penalty,
        weights = weights,
        offset = offset,
        intercept = intercept,
        control = control,
        nstages = nstages,
        nfolds = nfolds,
        stratified = stratified,
        alignment = alignment,
        moml_args = list(
            reward = reward,
            propensity_score = propensity_score
        )
    )
    ## refit if needed
    if (! isFALSE(refit) && length(res$et$selected) > 0) {
        if (isTRUE(refit)) {
            refit <- list(lambda = 1e-4, alignment = 1L)
        }
        idx <- res$et$selected
        ## inherit the penalty factors for those selected predictors
        if (! is.null(res$regularization$penalty_factor)) {
            refit$penalty_factor <- res$regularization$penalty_factor[idx]
        }
        refit_control <- modify_list(control, refit)
        refit_res <- .abclass(
            x = x[, idx, drop = FALSE],
            y = treatment,
            ## assume intercept, weights, loss are the same
            loss = loss,
            penalty = penalty,
            weights = weights,
            offset = offset,
            intercept = intercept,
            control = refit_control,
            ## cv
            nfolds = null0(refit$nfolds),
            stratified = null0(refit$stratified),
            alignment = null0(refit$alignment),
            ## et
            nstages = null0(refit$nstages),
            ## moml
            moml_args = list(
                reward = reward,
                propensity_score = propensity_score
            )
        )
        if (! is.null(refit_res$cross_validation)) {
            ## add cv idx
            cv_idx_list <- with(refit_res$cross_validation,
                                select_lambda(cv_accuracy_mean, cv_accuracy_sd))
            refit_res$cross_validation <- c(refit_res$cross_validation,
                                            cv_idx_list)
        }
        res$refit <- refit_res[! names(refit_res) %in% c("specs", "category")]
        res$refit$selected_coef <- idx
    } else {
        res$refit <- FALSE
    }
    ## add class
    class(res) <- c("et.moml", "moml", "abclass")
    ## return
    res
}
