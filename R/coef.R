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
##' @param relax_gamma A numeric value in \code{[0, 1]}, or a character value
##'     (\code{"cv_min"} or \code{"cv_1se"}), specifying how to blend the
##'     ET fit with its relaxed/debiased counterpart for an \code{et.abclass}
##'     object fitted with \code{relax} enabled (see \code{et.abclass()}).
##'     Only relevant when \code{relax} was enabled; ignored otherwise.  A
##'     numeric value is used directly as the blending weight.  \code{"cv_min"}
##'     and \code{"cv_1se"} select \code{relax_gamma} by cross-validation
##'     accuracy, analogous to \code{selection}, and require the object to have
##'     been fitted with \code{nfolds > 0}.  When \code{NULL} (the default),
##'     \code{relax_gamma} is chosen as \code{"cv_1se"} if cross-validation
##'     results are available, or \code{0} (the fully relaxed fit) otherwise.
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
                         selection = c("cv_1se", "cv_min", "all"),
                         relax_gamma = NULL,
                         ...)
{
    ## note: drop the dimension unless 'all' or multiple selected
    if (! (is.null(object$refit) || isFALSE(object$refit))) {
        tmp <- object$refit
        nlambda <- tmp$coefficients
        p <- nrow(object$coefficients) - as.integer(object$specs$intercept)
        dk <- dim(tmp$coefficients)[3L]
        coef_arr <- array(0, dim = c(dim(object$coefficients)[seq_len(2)], dk))
        idx <- object$refit$selected_coef
        if (object$specs$intercept) {
            idx <- c(1L, idx + 1L)
        }
        for (k in seq_len(dk)) {
            coef_arr[idx, , k] <- tmp$coefficients[, , k]
        }
        tmp$coefficients <- coef_arr
        return(coef.abclass(tmp, selection = selection, ...))
    }
    if (inherits(object, "et.abclass")) { # refit must be FALSE here
        beta <- object$coefficients[, , 1L, drop = TRUE]
        if (! isTRUE(object$et$relax)) {
            return(beta)
        }
        gamma_grid <- object$et$relax_gamma
        relax_coef <- object$et$relax_coefficients
        if (is.null(relax_coef)) {
            return(beta)
        }
        relax_beta <- relax_coef[, , 1L, drop = TRUE]
        cv <- object$cross_validation
        has_cv <- ! is.null(cv) &&
            length(cv$cv_accuracy_mean) == length(gamma_grid)
        if (is.numeric(relax_gamma)) {
            gamma_val <- relax_gamma[1L]
        } else if (has_cv) {
            gamma_sel <- if (is.null(relax_gamma)) {
                             "cv_1se"
                         } else {
                             match.arg(relax_gamma, c("cv_1se", "cv_min"))
                         }
            idx <- select_gamma(gamma_grid, cv$cv_accuracy_mean,
                                cv$cv_accuracy_sd)[[gamma_sel]]
            gamma_val <- gamma_grid[idx]
        } else if (is.null(relax_gamma)) {
            gamma_val <- 0
        } else {
            stop("No cross-validation results are available to select ",
                 "'relax_gamma' by \"", relax_gamma,
                 "\"; specify a numeric 'relax_gamma' instead.")
        }
        return(gamma_val * beta + (1 - gamma_val) * relax_beta)
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
            stop(sprintf("The 'selection' index must be <= %d.", dk))
        }
        ## do not drop dimension if multiple idx for binary classification
        return(object$coefficients[, , selection,
                                   drop = length(selection) == 1])
    }
    selection <- match.arg(selection, c("cv_1se", "cv_min", "all"))
    if (! length(object$cross_validation$cv_accuracy) || selection == "all") {
        return(object$coefficients)
    }
    cv_idx_list <- object$cross_validation
    selection_idx <- cv_idx_list[[selection]]
    object$coefficients[, , selection_idx, drop = TRUE]
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
                          selection = c("cv_1se", "cv_min", "all"),
                          ...)
{
    ## note: drop the dimension unless 'all' or multiple selected
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
            stop(sprintf("The 'selection' index must be <= %d.", dk))
        }
        return(object$coefficients[, , selection, drop = TRUE])
    }
    selection <- match.arg(selection, c("cv_1se", "cv_min", "all"))
    ## if selection is "all", return all coef
    if (selection == "all") {
        return(object$coefficients)
    }
    if (length(object$cross_validation$cv_accuracy) > 0L) {
        cv_idx_list <- object$cross_validation
        selection_idx <- cv_idx_list[[selection]]
        return(object$coefficients[, , selection_idx, drop = TRUE])
    }
    ## or use BIC for logistic model
    bic_vec <- BIC(object)
    if (! is.null(bic_vec)) {
        return(object$coefficients[, , which.min(bic_vec)])
    }
    ## or return all
    return(object$coefficients)
}
