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

##' Multi-Category Angle-Based Classification
##'
##' Multi-category angle-based large-margin classifiers with regularization by
##' the elastic-net or groupwise penalty.
##'
##' @name abclass
##'
##' @param x A numeric matrix representing the design matrix.  No missing valus
##'     are allowed.  The coefficient estimates for constant columns will be
##'     zero.  Thus, one should set the argument \code{intercept} to \code{TRUE}
##'     to include an intercept term instead of adding an all-one column to
##'     \code{x}.
##' @param y An integer vector, a character vector, or a factor vector
##'     representing the response label.
##' @param intercept A logical value indicating if an intercept should be
##'     considered in the model.  The default value is \code{TRUE} and the
##'     intercept is excluded from regularization.
##' @param weight A numeric vector for nonnegative observation weights. Equal
##'     observation weights are used by default.
##' @param loss A character value specifying the loss function.  The available
##'     options are \code{"logistic"} for the logistic deviance loss,
##'     \code{"boost"} for the exponential loss approximating Boosting machines,
##'     \code{"hinge-boost"} for hybrid of SVM and AdaBoost machine, and
##'     \code{"lum"} for largin-margin unified machines (LUM).  See Liu, et
##'     al. (2011) for details.
##' @param control A list of control parameters. See \code{abclass.control()}
##'     for details.
##' @param ... Other control parameters passed to \code{abclass.control()}.
##'
##' @return The function \code{abclass()} returns an object of class
##'     \code{abclass} representing a trained classifier; The function
##'     \code{abclass.control()} returns an object of class {abclass.control}
##'     representing a list of control parameters.
##'
##' @references
##'
##' Zhang, C., & Liu, Y. (2014). Multicategory Angle-Based Large-Margin
##' Classification. \emph{Biometrika}, 101(3), 625--640.
##'
##' Liu, Y., Zhang, H. H., & Wu, Y. (2011). Hard or soft classification?
##' large-margin unified machines. \emph{Journal of the American Statistical
##' Association}, 106(493), 166--177.
##'
##' @example inst/examples/ex-abclass.R
##'
##' @export
abclass <- function(x, y,
                    intercept = TRUE,
                    weight = NULL,
                    loss = c("logistic", "boost", "hinge-boost", "lum"),
                    control = list(),
                    ...)
{
    all_loss <- c("logistic", "boost", "hinge-boost", "lum")
    loss <- match.arg(as.character(loss), choices = all_loss)
    ## controls
    dot_list <- list(...)
    control <- do.call(abclass.control, modify_list(control, dot_list))
    res <- .abclass(
        x = x,
        y = y,
        intercept = intercept,
        weight = null2num0(weight),
        loss = loss,
        control = control
    )
    class(res) <- c("abclass_path", "abclass")
    ## return
    res
}


##' @rdname abclass
##'
##' @param lambda A numeric vector specifying the tuning parameter
##'     \emph{lambda}.  A data-driven \emph{lambda} sequence will be generated
##'     and used according to specified \code{alpha}, \code{nlambda} and
##'     \code{lambda_min_ratio} if this argument is left as \code{NULL} by
##'     default.  The specified \code{lambda} will be sorted in decreasing order
##'     internally and only the unique values will be kept.
##' @param alpha A numeric value in [0, 1] representing the mixing parameter
##'     \emph{alpha}.  The default value is \code{1.0}.
##' @param nlambda A positive integer specifying the length of the internally
##'     generated \emph{lambda} sequence.  This argument will be ignored if a
##'     valid \code{lambda} is specified.  The default value is \code{50}.
##' @param lambda_min_ratio A positive number specifying the ratio of the
##'     smallest lambda parameter to the largest lambda parameter.  The default
##'     value is set to \code{1e-4} if the sample size is larger than the number
##'     of predictors, and \code{1e-2} otherwise.
##' @param penalty_factor A numerical vector with nonnegative values specifying
##'     the adaptive penalty factors for individual predictors (excluding
##'     intercept).
##' @param grouped A logicial value.  Experimental flag to apply group
##'     penalties.
##' @param group_penalty A character vector specifying the name of the group
##'     penalty.
##' @param offset An optional numeric matrix for offsets of the decision
##'     functions.
##' @param kappa_ratio A positive number within (0, 1) specifying the ratio of
##'     reciprocal gamma parameter for group SCAD or group MCP.  A close-to-zero
##'     \code{kappa_ratio} would give a solution close to lasso solution.
##' @param lum_a A positive number greater than one representing the parameter
##'     \emph{a} in LUM, which will be used only if \code{loss = "lum"}.  The
##'     default value is \code{1.0}.
##' @param lum_c A nonnegative number specifying the parameter \emph{c} in LUM,
##'     which will be used only if \code{loss = "hinge-boost"} or \code{loss =
##'     "lum"}.  The default value is \code{1.0}.
##' @param boost_umin A negative number for adjusting the boosting loss for the
##'     internal majorization procedure.
##' @param maxit A positive integer specifying the maximum number of iteration.
##'     The default value is \code{10^4}.
##' @param epsilon A positive number specifying the relative tolerance that
##'     determines convergence.  The default value is \code{1e-7}.
##' @param standardize A logical value indicating if each column of the design
##'     matrix should be standardized internally to have mean zero and standard
##'     deviation equal to the sample size.  The default value is \code{TRUE}.
##'     Notice that the coefficient estimates are always returned on the
##'     original scale.
##' @param varying_active_set A logical value indicating if the active set
##'     should be updated after each cycle of coordinate-majorization-descent
##'     algorithm.  The default value is \code{TRUE} for usually more efficient
##'     estimation procedure.
##' @param verbose A nonnegative integer specifying if the estimation procedure
##'     is allowed to print out intermediate steps/results.  The default value
##'     is \code{0} for silent estimation procedure.
##'
##' @export
abclass.control <- function(lambda = NULL,
                            alpha = 1.0,
                            nlambda = 50L,
                            lambda_min_ratio = NULL,
                            penalty_factor = NULL,
                            grouped = TRUE,
                            group_penalty = c("lasso", "scad", "mcp"),
                            offset = NULL,
                            kappa_ratio = 0.9,
                            lum_a = 1.0,
                            lum_c = 0.0,
                            boost_umin = - 5.0,
                            maxit = 1e4L,
                            epsilon = 1e-7,
                            standardize = TRUE,
                            varying_active_set = TRUE,
                            verbose = 0L,
                            ...)
{
    group_penalty <-
        if (grouped) {
            match.arg(
                as.character(group_penalty),
                choices = c("lasso", "scad", "mcp")
            )
        } else {
            NULL
        }
    structure(list(
        alpha = alpha,
        lambda = lambda,
        nlambda = as.integer(nlambda),
        lambda_min_ratio = lambda_min_ratio,
        penalty_factor = penalty_factor,
        grouped = grouped,
        group_penalty = group_penalty,
        offset = offset,
        standardize = standardize,
        maxit = as.integer(maxit),
        epsilon = epsilon,
        varying_active_set = varying_active_set,
        verbose = as.integer(verbose),
        boost_umin = boost_umin,
        lum_a = lum_a,
        lum_c = lum_c,
        kappa_ratio = kappa_ratio
    ), class = "abclass.control")
}
