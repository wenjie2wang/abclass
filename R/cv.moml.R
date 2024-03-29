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

##' MOML with Cross-Validation
##'
##' Tune the regularization parameter for MOML by cross-validation.
##'
##' @inheritParams moml
##' @inheritParams cv.abclass
##'
##' @export
cv.moml <- function(x,
                    treatment,
                    reward,
                    propensity_score,
                    intercept = TRUE,
                    loss = c("logistic", "boost", "hinge-boost", "lum"),
                    control = moml.control(),
                    nfolds = 5L,
                    stratified = TRUE,
                    alignment = c("fraction", "lambda"),
                    refit = FALSE,
                    ...)
{
    ## nfolds
    nfolds <- as.integer(nfolds)
    if (nfolds < 3L) {
        stop("The 'nfolds' must be > 2.")
    }
    ## alignment
    if (is.numeric(alignment)) {
        alignment <- as.integer(alignment[1L])
    } else {
        all_alignment <- c("fraction", "lambda")
        alignment <- match.arg(alignment, choices = all_alignment)
        alignment <- match(alignment, all_alignment) - 1L
    }
    ## loss
    all_loss <- c("logistic", "boost", "hinge-boost", "lum")
    loss <- match.arg(loss, choices = all_loss)
    ## controls
    dot_list <- list(...)
    control <- do.call(moml.control, modify_list(control, dot_list))
    ## adjust lambda alignment
    if (alignment == 0L && length(control$lambda) > 0) {
        warning("Changed to `alignment` = 'lambda'",
                " for the specified lambda sequence.")
        alignment <- 1L
    }
    ## prepare arguments
    res <- .moml(x = x,
                 treatment = treatment,
                 reward = reward,
                 propensity_score = propensity_score,
                 intercept = intercept,
                 loss = loss,
                 control = control,
                 nfolds = nfolds,
                 stratified = stratified,
                 alignment = alignment)
    ## add cv idx
    cv_idx_list <- with(res$cross_validation,
                        select_lambda(cv_accuracy_mean, cv_accuracy_sd))
    res$cross_validation <- c(res$cross_validation, cv_idx_list)
    ## refit if needed
    if (! isFALSE(refit)) {
        if (isTRUE(refit)) {
            ## default controls
            refit <- list(lambda = 1e-6)
        }
        ## TODO allow selection of min and 1se
        coef_idx <- res$cross_validation$cv_1se
        idx <- which(apply(res$coefficients[- 1, , coef_idx] > 0, 1, any))
        ## inherit the penalty factors for those selected predictors
        if (! is.null(res$regularization$penalty_factor)) {
            refit$penalty_factor <- res$regularization$penalty_factor[idx]
        }
        refit_control <- modify_list(control, refit)
        refit_res <- .moml(
            x = x[, idx, drop = FALSE],
            treatment = treatment,
            reward = reward,
            propensity_score = propensity_score,
            ## assume intercept, weight, loss are the same with
            intercept = intercept,
            loss = loss,
            control = refit_control,
            nfolds = null0(refit$nfolds),
            stratified = ! isFALSE(refit$straitified),
            nstages = null0(refit$nstages)
        )
        if (! is.null(refit_res$cross_validation)) {
            ## add cv idx
            cv_idx_list <- with(refit_res$cross_validation,
                                select_lambda(cv_accuracy_mean, cv_accuracy_sd))
            refit_res$cross_validation <- c(refit_res$cross_validation,
                                            cv_idx_list)
        }
        res$refit <- refit_res[
            ! names(refit_res) %in% c("intercept", "loss", "category")
        ]
        res$refit$selected_coef <- idx
    } else {
        res$refit <- FALSE
    }
    ## add class
    class(res) <- c("cv.moml", "moml", "abclass")
    ## return
    res
}
