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
##' @param selection A character value specifying how to select a particular set
##'     of coefficient estimates from the entire solution path.  If the spcified
##'     \code{abclass} object contains the cross-validation results, one may set
##'     \code{selection} to \code{"cv_min"} (or \code{"cv_1se"}) for the
##'     estimates giving the smallest cross-validation error (or the set of
##'     estimates resulted from the largest \emph{lambda} within one standard
##'     error of the smallest cross-validation error).  The entire solution path
##'     will be returned in an array if \code{selection = "all"} or no
##'     cross-validation results are available in the input \code{abclass}
##'     object.
##' @param ... Other arguments not used now.
##'
##' @return A vector representing the predictions or an array representing the
##'     entire solution path.
##'
##' @importFrom stats coef
##' @export
coef.abclass <- function(object,
                         selection = c("cv_min", "cv_1se", "all"),
                         ...)
{
    ## set the selection index
    selection <- match.arg(selection, c("cv_min", "cv_1se", "all"))
    if (! length(object$cross_validation$cv_accuracy) || selection == "all") {
        return(object$coefficients)
    }
    ## cv_idx_list <- with(object$cross_validation,
    ##                     select_lambda(cv_accuracy_mean, cv_accuracy_sd))
    cv_idx_list <- object$cross_validation
    selection_idx <- cv_idx_list[[selection]]
    object$coefficients[, , selection_idx]
}
