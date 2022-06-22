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

##' Tune Angle-Based Classifiers by ET-Lasso
##'
##' Tune the regularization parameter for an angle-based large-margin classifier
##' by ET-Lasso (Yang, et al., 2019).
##'
##' @inheritParams abclass
##'
##' @param nstages A positive integer specifying for the number of stages in the
##'     ET-Lasso procedure.  By default, two rounds of tuning by random
##'     permutations will be performed as suggested in Yang, et al. (2019).
##'
##' @references
##'
##' Yang, S., Wen, J., Zhan, X., & Kifer, D. (2019). ET-Lasso: A new efficient
##' tuning of lasso-type regularization for high-dimensional data. In
##' /emph{Proceedings of the 25th ACM SIGKDD International Conference on
##' Knowledge Discovery \& Data Mining} (pp. 607--616).
##'
##' @export
et.abclass <- function(x, y,
                       intercept = TRUE,
                       weight = NULL,
                       loss = c("logistic", "boost", "hinge-boost", "lum"),
                       control = list(),
                       nstages = 2,
                       ...)
{
    ## nstages
    nstages <- as.integer(nstages)
    if (nstages < 1L) {
        stop("The 'nstages' must be a positive integer.")
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
             nstages = nstages,
             main_fit = FALSE),
        control
    )
    args_to_call <- args_to_call[
        names(args_to_call) %in% formal_names(.abclass)
    ]
    res <- do.call(.abclass, args_to_call)
    ## add class
    class_suffix <- if (control$grouped)
                        paste0("_group_", control$group_penalty)
                    else
                        "_net"
    res_cls <- paste0(loss2, class_suffix)
    class(res) <- c(res_cls, "et.abclass", "abclass")
    ## return
    res
}
