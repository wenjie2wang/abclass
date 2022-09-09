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

##' Coefficient Estimates of A Trained Angle-Based Classifier
##'
##' Extract coefficient estimates from an \code{abclass} object.
##'
##' @param object An object of class \code{abclass}.
##' @param selection An integer vector for the indices of solution path or a
##'     character value specifying how to select a particular set of coefficient
##'     estimates from the entire solution path.  If the specified
##'     \code{abclass} object contains the cross-validation results, one may set
##'     \code{selection} to \code{"cv_min"} (or \code{"cv_1se"}) for the
##'     estimates giving the smallest cross-validation error (or the set of
##'     estimates resulted from the largest \emph{lambda} within one standard
##'     error of the smallest cross-validation error).  The entire solution path
##'     will be returned in an array if \code{selection = "all"} or no
##'     cross-validation results are available in the specified \code{abclass}
##'     object.
##' @param ... Other arguments not used now.
##'
##' @return A matrix representing the coefficient estimates or an array
##'     representing all the selected solutions.
##'
##' @examples
##' ## see examples of `abclass()`.
##'
##' @importFrom stats coef
##' @export
coef.abclass <- function(object,
                         selection = c("cv_min", "cv_1se", "all"),
                         ...)
{
    if (inherits(object, "et.abclass")) {
        if (isFALSE(object$et$refit)) {
            return(object$coefficients)
        } else {
            tmp <- object$et$refit
            nlambda <- tmp$coefficients
            p <- nrow(object$coefficients) - as.integer(object$intercept)
            dk <- dim(tmp$coefficients)[3L]
            coef_arr <- array(0, dim = c(dim(object$coefficients), dk))
            idx <- object$et$selected
            if (object$intercept) {
                idx <- c(1L, idx + 1L)
            }
            for (k in seq_len(dk)) {
                coef_arr[idx, , k] <- tmp$coefficients[, , k]
            }
            tmp$coefficients <- coef_arr
            return(coef.abclass(tmp, selection = selection, ...))
        }
    }
    ## if only one solution
    dim_coef <- dim(object$coefficients)
    dk <- dim_coef[3L]
    if (is.na(dk)) {
        return(object$coefficients)
    }
    if (dk == 1L) {
        return(object$coefficients[, , 1L, drop = TRUE])
    }
    ## for integer indices
    if (is.numeric(selection)) {
        selection <- as.integer(selection)
        if (any(selection > dk)) {
            stop(sprintf("The integer 'selection' must <= %d.", dk))
        }
        return(object$coefficients[, , selection])
    }
    selection <- match.arg(selection, c("cv_min", "cv_1se", "all"))
    if (! length(object$cross_validation$cv_accuracy) || selection == "all") {
        return(object$coefficients)
    }
    cv_idx_list <- object$cross_validation
    selection_idx <- cv_idx_list[[selection]]
    object$coefficients[, , selection_idx]
}


##' Coefficient Estimates of A Trained Sup-Norm Classifier
##'
##' Extract coefficient estimates from an \code{supclass} object.
##'
##' @param object An object of class \code{supclass}.
##' @param selection An integer vector for the indices of solution or a
##'     character value specifying how to select a particular set of coefficient
##'     estimates from the entire solution path.  If the specified
##'     \code{supclass} object contains the cross-validation results, one may
##'     set \code{selection} to \code{"cv_min"} (or \code{"cv_1se"}) for the
##'     estimates giving the smallest cross-validation error (or the set of
##'     estimates resulted from the largest \emph{lambda} within one standard
##'     error of the smallest cross-validation error).  The entire solution path
##'     will be returned in an array if \code{selection = "all"} or no
##'     cross-validation results are available in the specified \code{supclass}
##'     object.
##' @param ... Other arguments not used now.
##'
##' @return A matrix representing the coefficient estimates or an array
##'     representing all the selected solutions.
##'
##' @examples
##' ## see examples of `supclass()`.
##'
##' @importFrom stats coef
##' @export
coef.supclass <- function(object,
                          selection = c("cv_min", "cv_1se", "all"),
                          ...)
{
    ## if only one solution
    dim_coef <- dim(object$coefficients)
    dk <- dim_coef[3L]
    if (is.na(dk)) {
        return(object$coefficients)
    }
    if (dk == 1L) {
        return(object$coefficients[, , 1L, drop = TRUE])
    }
    ## for integer indices
    if (is.numeric(selection)) {
        selection <- as.integer(selection)
        if (any(selection > dk)) {
            stop(sprintf("The integer 'selection' must <= %d.", dk))
        }
        return(object$coefficients[, , selection])
    }
    selection <- match.arg(selection, c("cv_min", "cv_1se", "all"))
    ## BIC for logistic model
    bic_vec <- BIC(object)
    if (! is.null(bic_vec)) {
        return(object$coefficients[, , which.min(bic_vec)])
    }
    ## otherwise
    if (! length(object$cross_validation$cv_accuracy) || selection == "all") {
        return(object$coefficients)
    }
    cv_idx_list <- object$cross_validation
    selection_idx <- cv_idx_list[[selection]]
    object$coefficients[, , selection_idx]
}
