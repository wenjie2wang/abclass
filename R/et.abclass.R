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
##' by the ET-Lasso method (Yang, et al., 2019).
##'
##' @inheritParams abclass
##'
##' @param nstages A positive integer specifying for the number of stages in the
##'     ET-Lasso procedure.  By default, two rounds of tuning by random
##'     permutations will be performed as suggested in Yang, et al. (2019).
##' @param refit A logical value indicating if a new classifier should be
##'     trained using the selected predictors.  This argument can also be a list
##'     with named elements, which will be passed to \code{abclass.control()} to
##'     specify how the new classifier should be trained.
##'
##' @references
##'
##' Yang, S., Wen, J., Zhan, X., & Kifer, D. (2019). ET-Lasso: A new efficient
##' tuning of lasso-type regularization for high-dimensional data. In
##' \emph{Proceedings of the 25th ACM SIGKDD International Conference on
##' Knowledge Discovery & Data Mining} (pp. 607--616).
##'
##' @export
et.abclass <- function(x, y,
                       intercept = TRUE,
                       weight = NULL,
                       loss = c("logistic", "boost", "hinge-boost", "lum"),
                       control = list(),
                       nstages = 2,
                       refit = list(lambda = 1e-6),
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
    ## refit if needed
    if (! isFALSE(refit) && length(res$et$selected) > 0) {
        if (isTRUE(refit)) {
            refit <- list(lambda = 1e-6)
        }
        idx <- res$et$selected
        ## inherit the group weights for those selected predictors
        if (! is.null(res$regularization$group_weight)) {
            refit$group_weight <- res$regularization$group_weight[idx]
        }
        refit_control <- modify_list(control, refit)
        args_to_call <- c(
            list(x = x[, idx, drop = FALSE],
                 y = y,
                 ## assume intercept, weight, loss are the same with et-lasso
                 intercept = intercept,
                 weight = res$weight,
                 loss = loss2,
                 nstages = 0,
                 main_fit = TRUE),
            refit_control
        )
        args_to_call <- args_to_call[
            names(args_to_call) %in% formal_names(.abclass)
        ]
        refit_res <- do.call(.abclass, args_to_call)
        if (! is.null(refit_res$cross_validation)) {
            ## add cv idx
            cv_idx_list <- with(refit_res$cross_validation,
                                select_lambda(cv_accuracy_mean, cv_accuracy_sd))
            refit_res$cross_validation <- c(refit_res$cross_validation,
                                            cv_idx_list)
        }
        res$refit <- refit_res[! names(refit_res) %in%
                               c("intercept", "weight", "loss", "category")]
        res$refit$selected_coef <- idx
    } else {
        res$refit <- FALSE
    }
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
