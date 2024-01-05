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

##' Multi-Category Outcome-Weighted Margin-Based Learning (MOML)
##'
##' Performs the outcome-weighted margin-based learning for multicategory
##' treatments proposed by Zhang, et al. (2020).
##'
##' @inheritParams abclass_propscore
##' @param reward A numeric vector representing the rewards.  It is assumed that
##'     a larger reward is more desirable.
##' @param propensity_score A numeric vector taking values between 0 and 1
##'     representing the propensity score.
##' @param ... Other arguments passed to the control function, which calls the
##'     \code{abclass.control()} internally.
##'
##' @references
##'
##' Zhang, C., Chen, J., Fu, H., He, X., Zhao, Y., & Liu, Y. (2020).
##' Multicategory outcome weighted margin-based learning for estimating
##' individualized treatment rules. Statistica Sinica, 30, 1857--1879.
##'
##' @export
moml <- function(x,
                 treatment,
                 reward,
                 propensity_score,
                 intercept = TRUE,
                 loss = c("logistic", "boost", "hinge-boost", "lum"),
                 control = moml.control(),
                 ...)
{
    all_loss <- c("logistic", "boost", "hinge-boost", "lum")
    loss <- match.arg(as.character(loss), choices = all_loss)
    ## controls
    dot_list <- list(...)
    control <- do.call(moml.control, modify_list(control, dot_list))
    res <- .moml(
        x = x,
        treatment = treatment,
        reward = reward,
        propensity_score = propensity_score,
        intercept = intercept,
        loss = loss,
        control = control
    )
    class(res) <- c("moml_path", "moml", "abclass")
    ## return
    res
}


##' @rdname moml
moml.control <- function(...)
{
    ctrl <- abclass.control(...)
    class(ctrl) <- "moml.control"
    ctrl
}
