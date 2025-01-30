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

##' Tune Angle-Based Classifiers by Cross-Validation
##'
##' Tune the regularization parameter for an angle-based large-margin classifier
##' by cross-validation.
##'
##' @inheritParams abclass
##'
##' @param nfolds A positive integer specifying the number of folds for
##'     cross-validation.  Five-folds cross-validation will be used by default.
##'     An error will be thrown out if the \code{nfolds} is specified to be less
##'     than 2.
##' @param stratified A logical value indicating if the cross-validation
##'     procedure should be stratified by the response label. The default value
##'     is \code{TRUE} to ensure the same number of categories be used in
##'     validation and training.
##' @param alignment A character vector specifying how to align the lambda
##'     sequence used in the main fit with the cross-validation fits.  The
##'     available options are \code{"fraction"} for allowing cross-validation
##'     fits to have their own lambda sequences and \code{"lambda"} for using
##'     the same lambda sequence of the main fit.  The option \code{"lambda"}
##'     will be applied if a meaningful \code{lambda} is specified.  The default
##'     value is \code{"fraction"}.
##' @param refit A logical value indicating if a new classifier should be
##'     trained using the selected predictors or a named list that will be
##'     passed to \code{abclass.control()} to specify how the new classifier
##'     should be trained.
##'
##' @return An S3 object of class \code{cv.abclass} and \code{abclass}.
##'
##' @export
cv.abclass <- function(x, y,
                       loss = c("logistic", "boost", "hinge.boost", "lum"),
                       penalty = c("glasso", "lasso"),
                       weights = NULL,
                       offset = NULL,
                       intercept = TRUE,
                       control = list(),
                       nfolds = 5L,
                       stratified = TRUE,
                       alignment = c("fraction", "lambda"),
                       refit = FALSE,
                       ...)
{
    loss <- match.arg(as.character(loss)[1],
                      choices = .all_abclass_losses)
    penalty <- match.arg(as.character(penalty)[1],
                         choices = .all_abclass_penalties)
    ## nfolds
    nfolds <- as.integer(nfolds)
    if (nfolds < 3L) {
        stop("The 'nfolds' must be > 2.")
    }
    ## controls
    dot_list <- list(...)
    control <- do.call(abclass.control, modify_list(control, dot_list))
    ## prepare arguments
    res <- .abclass(
        x = x,
        y = y,
        loss = loss,
        penalty = penalty,
        weights = weights,
        offset = offset,
        intercept = intercept,
        control = control,
        nfolds = nfolds,
        stratified = stratified,
        alignment = alignment
    )
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
        refit_res <- .abclass(
            x = x[, idx, drop = FALSE],
            y = y,
            ## assume intercept, weights, loss are the same with
            loss = loss,
            penalty = penalty,
            weights = weights,
            offset = offset,
            intercept = intercept,
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
            ! names(refit_res) %in%
            c("category", "loss", "penalty", "weights", "offset", "intercept")
        ]
        res$refit$selected_coef <- idx
    } else {
        res$refit <- FALSE
    }
    ## add class
    class(res) <- c("cv.abclass", "abclass")
    ## return
    res
}
