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

abrank <- function(x, y, qid,
                   weight = NULL,
                   loss = c("logistic", "boost", "hinge-boost", "lum"),
                   control = list(),
                   ...)
{
    all_loss <- c("logistic", "boost", "hinge-boost", "lum")
    loss <- match.arg(as.character(loss), choices = all_loss)
    ## controls
    dot_list <- list(...)
    control <- do.call(abrank.control, modify_list(control, dot_list))
    res <- .abrank(
        x = x,
        y = y,
        qid = qid,
        weight = null2num0(weight),
        loss = loss,
        control = control
    )
    class(res) <- c("abrank_path", "abrank")
    ## return
    res
}

abrank.control <- function(lambda = NULL,
                           alpha = 1.0,
                           nlambda = 50L,
                           lambda_min_ratio = NULL,
                           offset = NULL,
                           query_weight = FALSE,
                           delta_weight = FALSE,
                           delta_adaptive = FALSE,
                           delta_maxit = 10,
                           lum_a = 1.0,
                           lum_c = 1.0,
                           boost_umin = - 5.0,
                           maxit = 1e5L,
                           epsilon = 1e-4,
                           standardize = TRUE,
                           varying_active_set = TRUE,
                           verbose = 0L,
                           ...)
{
    structure(list(
        alpha = alpha,
        lambda = null2num0(lambda),
        nlambda = as.integer(nlambda),
        lambda_min_ratio = lambda_min_ratio,
        offset = null2num0(offset),
        query_weight = query_weight,
        delta_weight = delta_weight,
        delta_adaptive = delta_adaptive,
        delta_maxit = delta_maxit,
        standardize = standardize,
        maxit = as.integer(maxit),
        epsilon = epsilon,
        varying_active_set = varying_active_set,
        verbose = as.integer(verbose),
        boost_umin = boost_umin,
        lum_a = lum_a,
        lum_c = lum_c
    ), class = "abrank.control")
}
