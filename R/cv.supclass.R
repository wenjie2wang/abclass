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

##' Tune Sup-Norm Classifiers by Cross-Validation
##'
##' Tune the regularization parameter lambda for a sup-norm classifier by
##' cross-validation.
##'
##' @inheritParams supclass
##'
##' @param nfolds A positive integer specifying the number of folds for
##'     cross-validation.  Five-folds cross-validation will be used by default.
##'     An error will be thrown out if the \code{nfolds} is specified to be less
##'     than 2.
##' @param stratified A logical value indicating if the cross-validation
##'     procedure should be stratified by the response label. The default value
##'     is \code{TRUE} to ensure the same number of categories be used in
##'     validation and training.
##' @param ... Other arguments passed to \code{supclass}.
##'
##' @return An S3 object of class \code{cv.supclass}.
##'
##' @importFrom stats sd
##' @importFrom parallel mclapply
##' @export
cv.supclass <- function(x, y,
                        model = c("logistic", "psvm", "svm"),
                        penalty = c("lasso", "scad"),
                        start = NULL,
                        control = list(),
                        nfolds = 5L,
                        stratified = TRUE,
                        ...)
{
    ## nfolds
    nfolds <- as.integer(nfolds)
    if (nfolds < 3L) {
        stop("The 'nfolds' must be > 2.")
    }
    ## preprocess
    cat_y <- cat2z(y, zero_based = FALSE)
    cv_list <- cv_samples(
        nobs = length(y),
        nfolds = nfolds,
        strata = if (stratified) cat_y$y - 1L else { integer() }
    )
    ## main fit
    res <- supclass(
        x = x,
        y = y,
        model = model,
        penalty = penalty,
        start = start,
        control = control,
        ...
    )
    nlambda <- length(res$regularization$lambda)
    ## cv part
    cv_res <- parallel::mclapply(seq_len(nfolds),
                                 function(i) {
                                     train_idx <- cv_list$train_index[[i]]
                                     valid_idx <- cv_list$valid_index[[i]]
                                     tmp_fit <- supclass(
                                         x = x[train_idx, , drop = FALSE],
                                         y = y[train_idx],
                                         model = model,
                                         penalty = penalty,
                                         start = start,
                                         control = control,
                                         ...
                                     )
                                     valid_pred <- predict(
                                         object = tmp_fit,
                                         newx = x[valid_idx, , drop = FALSE],
                                         type = "class",
                                         selection = "all"
                                     )
                                     if (nlambda > 1) {
                                         sapply(
                                             valid_pred,
                                             function(a) mean(a == y[valid_idx])
                                         )
                                     } else {
                                         mean(valid_pred == y[valid_idx])
                                     }
                                 })
    ## aggregate cv results
    cv_res <- do.call(cbind, cv_res)
    res$cross_validation <- list(
        nfolds = nfolds,
        stratified = TRUE,
        cv_accuracy = cv_res,
        cv_accuracy_mean = rowMeans(cv_res),
        cv_accuracy_sd = apply(cv_res, 1L, sd)
    )
    cv_res0 <- with(res$cross_validation,
                    select_lambda(cv_accuracy_mean, cv_accuracy_sd))
    res$cross_validation <- c(res$cross_validation, cv_res0)
    ## add class
    class(res) <- c(sprintf("supclass_%s_%s", model, penalty),
                    "cv.supclass", "supclass")
    ## return
    res
}
