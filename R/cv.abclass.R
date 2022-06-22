##
## R package abclass developed by Wenjie Wang <wang@wwenjie.org>
## Copyright (C) 2021-2022 Eli Lilly and Company
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
##'
##' @return An S3 object of class \code{cv.abclass}.
##'
##' @export
cv.abclass <- function(x, y,
                       intercept = TRUE,
                       weight = NULL,
                       loss = c("logistic", "boost", "hinge-boost", "lum"),
                       control = list(),
                       nfolds = 5L,
                       stratified = TRUE,
                       alignment = c("fraction", "lambda"),
                       ...)
{
    ## nfolds
    nfolds <- as.integer(nfolds)
    if (nfolds < 2L) {
        stop("The 'nfolds' must be > 1.")
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
    loss2 <- gsub("-", "_", loss, fixed = TRUE)
    ## controls
    dot_list <- list(...)
    control <- do.call(abclass.control, modify_list(control, dot_list))
    ## prepare arguments
    args_to_call <- c(
        list(x = x,
             y = y,
             intercept = intercept,
             weight = null2num0(weight),
             loss = loss2,
             nfolds = nfolds,
             stratified = stratified,
             alignment = alignment,
             main_fit = TRUE),
        control
    )
    args_to_call <- args_to_call[
        names(args_to_call) %in% formal_names(.abclass)
    ]
    res <- do.call(.abclass, args_to_call)
    ## add cv idx
    cv_idx_list <- with(res$cross_validation,
                        select_lambda(cv_accuracy_mean, cv_accuracy_sd))
    res$cross_validation <- c(res$cross_validation, cv_idx_list)
    ## add class
    class_suffix <- if (control$grouped)
                        paste0("_group_", control$group_penalty)
                    else
                        "_net"
    res_cls <- paste0(loss2, class_suffix)
    class(res) <- c(res_cls, "cv.abclass", "abclass")
    ## return
    res
}
