##
## R package abclass developed by Wenjie Wang <wang@wwenjie.org>
## Copyright (C) 2021-2023 Eli Lilly and Company
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

##' Estimate Propensity Score by the Angle-Based Classifiers
##'
##' A wrap function to estimate the propensity score by the multi-category
##' angle-based large-margin classifiers.
##'
##' @inheritParams abclass
##'
##' @param treatment The assigned treatments represented by a character,
##'     integer, numeric, or factor vector.
##' @param tuning A character vector specifying the tuning method.  This
##'     argument will be ignored if a single \code{lambda} is specified through
##'     \code{control}.
##' @param ... Other arguments passed to the corresponding methods.
##'
##' @export
abclass_propscore <-
    function(x,
             treatment,
             intercept = TRUE,
             weight = NULL,
             loss = c("logistic", "boost", "hinge-boost", "lum"),
             control = list(),
             tuning = c("et", "cv"),
             ...)
{
    all_loss <- c("logistic", "boost", "hinge-boost", "lum")
    loss <- match.arg(as.character(loss), choices = all_loss)
    tuning <- match.arg(as.character(tuning), choices = c("et", "cv"))
    dot_list <- list(...)
    control <- do.call(abclass.control, modify_list(control, dot_list))
    call_list <- list(
        x = x,
        y = treatment,
        intercept = intercept,
        weight = weight,
        loss = loss,
        control = control,
        ...
    )
    res <- if (length(control$lambda) == 1) {
               do.call(abclass, call_list)
           } else if (tuning == "et") {
               do.call(et.abclass, call_list)
           } else {
               do.call(cv.abclass, call_list)
           }
    idx_mat <- cbind(seq_along(treatment), res$category$y + 1L)
    prob_est <- predict(res, newx = x, type = "probability")
    prob_est <- prob_est[idx_mat]
    attr(prob_est, "model") <- res
    prob_est
}
