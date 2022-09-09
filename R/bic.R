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

##' @importFrom stats BIC
##' @export
BIC.logistic_suplasso <- function(object, ...)
{
    nll <- attr(object$coefficients, "negLogL")
    k <- apply(object$coefficients, 3L, function(a) {
        sum(abs(a) > .Machine$double.eps)
    })
    nobs <- length(object$category$y)
    log(nobs) * k + 2 * nll * nobs
}

##' @export
BIC.logistic_supscad <- BIC.logistic_suplasso

##' @export
BIC.supclass <- function(object, ...)
{
    NULL
}
