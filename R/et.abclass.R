##
## R package abclass developed by Wenjie Wang <wang@wwenjie.org>
## Copyright (C) 2021-2026 Eli Lilly and Company
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

##' Tune Angle-Based Classifiers by ET-Lasso
##'
##' Tune the regularization parameter for an angle-based large-margin classifier
##' by the ET-Lasso method (Yang, et al., 2019).
##'
##' The ET-Lasso procedure is intended for tuning the \code{lambda} parameter
##' solely.  The arguments regarding cross-validation, \code{nfolds},
##' \code{stratified}, and \code{alignment}, allow one to estimate the
##' prediction accuracy by cross-validation for the model estimates resulted
##' from the ET-Lasso procedure, which can be helpful for one to choose other
##' tuning parameters (e.g., \code{alpha}).
##'
##' @inheritParams cv.abclass
##'
##' @param nstages A positive integer specifying for the number of stages in the
##'     ET-Lasso procedure.  By default, two rounds of tuning by random
##'     permutations will be performed as suggested in Yang, et al. (2019).
##' @param relax A logical value or a list specifying whether and how to
##'     obtain a relaxed-lasso-style debiased fit on the ET-selected support,
##'     mirroring \pkg{glmnet}'s \code{relax = TRUE}.  The default,
##'     \code{FALSE}, disables it.  \code{TRUE} enables it with the default
##'     \code{gamma} grid \code{seq(0, 1, length.out = 11)} and
##'     \code{lambda = 1e-6}.  A list may specify either or both of
##'     \code{gamma} (a numeric vector of blending weights in \code{[0, 1]},
##'     where \code{1} recovers the plain ET fit and \code{0} the fully
##'     relaxed/debiased fit) and \code{lambda} (the near-zero regularization
##'     used for the debiased refit on the ET-selected support).  When
##'     \code{nfolds > 0}, the cross-validation accuracy is reported for
##'     every \code{gamma} in the grid, and \code{coef.abclass()}/
##'     \code{predict.abclass()} can select among them via their own
##'     \code{gamma} argument.
##'
##' @return An S3 object of class \code{et.abclass} and \code{abclass}.
##'
##' @references
##'
##' Yang, S., Wen, J., Zhan, X., & Kifer, D. (2019). ET-Lasso: A new efficient
##' tuning of lasso-type regularization for high-dimensional data. In
##' \emph{Proceedings of the 25th ACM SIGKDD International Conference on
##' Knowledge Discovery & Data Mining} (pp. 607--616).
##'
##' @export
et.abclass <- function(x, y,
                       loss = c("logistic", "boost", "hinge.boost", "lum"),
                       penalty = c("glasso", "lasso"),
                       weights = NULL,
                       offset = NULL,
                       intercept = TRUE,
                       control = list(),
                       nstages = 2L,
                       nfolds = 0L,
                       stratified = TRUE,
                       alignment = c("fraction", "lambda"),
                       refit = FALSE,
                       relax = FALSE,
                       ...)
{
    loss <- match.arg(as.character(loss)[1],
                      choices = .all_abclass_losses)
    penalty <- match.arg(as.character(penalty)[1],
                         choices = .all_abclass_penalties)
    ## nstages
    nstages <- as.integer(nstages)
    if (nstages < 1L) {
        stop("The 'nstages' must be a positive integer.")
    }
    ## relax
    if (isFALSE(relax)) {
        do_relax <- FALSE
        relax_lambda <- 1e-6
        relax_gamma <- seq(0, 1, length.out = 11)
    } else {
        if (isTRUE(relax)) {
            relax <- list()
        }
        if (! is.list(relax)) {
            stop("The 'relax' must be 'TRUE', 'FALSE', or a list.")
        }
        do_relax <- TRUE
        relax_lambda <- if (is.null(relax$lambda)) 1e-6 else relax$lambda
        relax_gamma <- if (is.null(relax$gamma)) {
                              seq(0, 1, length.out = 11)
                          } else {
                              relax$gamma
                          }
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
        nstages = nstages,
        nfolds = nfolds,
        stratified = stratified,
        alignment = alignment,
        relax = do_relax,
        relax_lambda = relax_lambda,
        relax_gamma = relax_gamma
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
            y = y,
            ## assume intercept, weights, loss are the same with et-lasso
            loss = loss,
            penalty = penalty,
            intercept = intercept,
            weights = res$weights,
            offset = res$offset,
            control = refit_control,
            ## cv
            nfolds = null0(refit$nfolds),
            stratified = ! isFALSE(refit$straitified),
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
                               c("category", "loss", "penalty",
                                 "weights", "offset", "intercept")]
        res$refit$selected_coef <- idx
    } else {
        res$refit <- FALSE
    }
    ## add class
    class(res) <- c("et.abclass", "abclass")
    ## return
    res
}
