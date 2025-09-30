##
## R package abclass developed by Wenjie Wang <wang@wwenjie.org>
## Copyright (C) 2021-2025 Eli Lilly and Company
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
##' @param loss A character value specifying the loss function.  The available
##'     options are \code{"logistic"} for the logistic deviance loss,
##'     \code{"boost"} for the exponential loss approximating Boosting machines,
##'     \code{"hinge.boost"} for hybrid of SVM and AdaBoost machine, and
##'     \code{"lum"} for largin-margin unified machines (LUM).  See Liu, et
##'     al. (2011) for details.
##' @param penalty A character vector specifying the name of the penalty.
##' @param weights A numeric vector for nonnegative observation weights. Equal
##'     observation weights are used by default.
##' @param offset An optional numeric matrix for offsets of the decision
##'     functions.
##' @param intercept A logical value indicating if an intercept should be
##'     considered in the model.  The default value is \code{TRUE} and the
##'     intercept is excluded from regularization.
##' @param control A list of control parameters. See \code{abclass.control()}
##'     for details.
##' @param ... Other control parameters passed to \code{abclass.control()}.
##'
##' @return The function \code{abclass()} returns an object of class
##'     \code{abclass} representing a trained classifier; The function
##'     \code{abclass.control()} returns an object of class
##'     \code{abclass.control} representing a list of control parameters.
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
                    loss = c("logistic", "boost", "hinge.boost", "lum"),
                    penalty = c("glasso", "lasso"),
                    weights = NULL,
                    offset = NULL,
                    intercept = TRUE,
                    control = list(),
                    ...)
{
    loss <- match.arg(as.character(loss)[1],
                      choices = .all_abclass_losses)
    penalty <- match.arg(as.character(penalty)[1],
                         choices = .all_abclass_penalties)
    ## controls
    dot_list <- list(...)
    control <- do.call(abclass.control, modify_list(control, dot_list))
    res <- .abclass(
        x = x,
        y = y,
        loss = loss,
        penalty = penalty,
        weights = weights,
        offset = offset,
        intercept = intercept,
        control = control
    )
    class(res) <- c("abclass_path", "abclass")
    if (isTRUE(control$save_call)) {
        res$call <- match.call()
    }
    ## return
    res
}


##' @rdname abclass
##'
##' @param lum_a A positive number greater than one representing the parameter
##'     \emph{a} in LUM, which will be used only if \code{loss = "lum"}.  The
##'     default value is \code{1.0}.
##' @param lum_c A nonnegative number specifying the parameter \emph{c} in LUM,
##'     which will be used only if \code{loss = "hinge.boost"} or \code{loss =
##'     "lum"}.  The default value is \code{1.0}.
##' @param boost_umin A negative number for adjusting the boosting loss for the
##'     internal majorization procedure.
##' @param alpha A numeric value in $[0,1]$ representing the mixing parameter
##'     \emph{alpha}.  The default value is \code{1.0}.
##' @param nlambda A positive integer specifying the length of the internally
##'     generated \emph{lambda} sequence.  This argument will be ignored if a
##'     valid \code{lambda} is specified.  The default value is \code{50}.
##' @param lambda A numeric vector specifying the tuning parameter
##'     \emph{lambda}.  A data-driven \emph{lambda} sequence will be generated
##'     and used according to specified \code{alpha}, \code{nlambda} and
##'     \code{lambda_min_ratio} if this argument is left as \code{NULL} by
##'     default.  The specified \code{lambda} will be sorted in decreasing order
##'     internally and only the unique values will be kept.
##' @param lambda_min_ratio A positive number specifying the ratio of the
##'     smallest lambda parameter to the largest lambda parameter.  The default
##'     value is set to \code{1e-4} if the sample size is larger than the number
##'     of predictors, and \code{1e-2} otherwise.
##' @param lambda_max_alpha_min A positive number specifying the minimum
##'     denominator when the function determines the largest lambda.  If the
##'     \code{lambda} is not specified, the largest lambda will be determined by
##'     the data and be the large enough lambda (that would result in all zero
##'     estimates for the covariates with positive penalty factors) divided by
##'     \code{max(alpha, lambda_max_alpha_min)}.
##' @param penalty_factor A numerical vector with nonnegative values specifying
##'     the adaptive penalty factors for individual predictors (excluding
##'     intercept).
##' @param ncv_kappa A positive number within $(0,1)$ specifying the ratio of
##'     reciprocal gamma parameter for group SCAD or group MCP.  A close-to-zero
##'     \code{ncv_kappa} would give a solution close to lasso solution.
##' @param gel_tau A positive parameter tau for group exponential lasso penalty.
##' @param mellowmax_omega A positive parameter omega for Mellowmax penalty.  It
##'     is experimental and subject to removal in future.
##' @param lower_limit,upper_limit Numeric matrices representing the desired
##'     lower and upper limits for the coefficient estimates, respectively.
##' @param epsilon A positive number specifying the relative tolerance that
##'     determines convergence.
##' @param maxit A positive integer specifying the maximum number of iteration.
##' @param standardize A logical value indicating if each column of the design
##'     matrix should be standardized internally to have mean zero and standard
##'     deviation equal to the sample size.  The default value is \code{TRUE}.
##'     Notice that the coefficient estimates are always returned on the
##'     original scale.
##' @param varying_active_set A logical value indicating if the active set
##'     should be updated after each cycle of coordinate-descent algorithm.  The
##'     default value is \code{TRUE} for usually more efficient estimation
##'     procedure.
##' @param adjust_mm An experimental logical value specifying if the estimation
##'     procedure should track loss function and adjust the MM lower bound if
##'     needed.
##' @param save_call A logical value indicating if the function call of the
##'     model fitting should be saved.  If \code{TRUE}, the function call will
##'     be saved in the returned object so that one can utilize
##'     \code{stats::update()} to update the argument specifications
##'     conveniently.
##' @param verbose A nonnegative integer specifying if the estimation procedure
##'     is allowed to print out intermediate steps/results.  The default value
##'     is \code{0} for silent estimation procedure.
##'
##' @export
abclass.control <- function(## loss
                            lum_a = 1.0,
                            lum_c = 0.0,
                            boost_umin = -5.0,
                            ## penalty
                            alpha = 1.0,
                            lambda = NULL,
                            nlambda = 50L,
                            lambda_min_ratio = NULL,
                            lambda_max_alpha_min = 0.01,
                            penalty_factor = NULL,
                            ncv_kappa = 0.1,
                            gel_tau = 0.33,
                            mellowmax_omega = 1,
                            ## coef
                            lower_limit = - Inf,
                            upper_limit = Inf,
                            ## optim
                            epsilon = 1e-7,
                            maxit = 1e5L,
                            standardize = TRUE,
                            varying_active_set = TRUE,
                            adjust_mm = FALSE,
                            ## misc
                            save_call = FALSE,
                            verbose = 0L)
{
    ## TODO validate data types
    structure(list(
        lum_a = lum_a,
        lum_c = lum_c,
        boost_umin = boost_umin,
        alpha = alpha,
        lambda = lambda,
        nlambda = as.integer(nlambda),
        lambda_min_ratio = lambda_min_ratio,
        lambda_max_alpha_min = lambda_max_alpha_min,
        penalty_factor = penalty_factor,
        ncv_kappa = ncv_kappa,
        gel_tau = gel_tau,
        mellowmax_omega = mellowmax_omega,
        lower_limit = as.matrix(lower_limit),
        upper_limit = as.matrix(upper_limit),
        epsilon = epsilon,
        maxit = as.integer(maxit),
        standardize = standardize,
        varying_active_set = varying_active_set,
        adjust_mm = adjust_mm,
        save_call = save_call,
        verbose = as.integer(verbose)
    ), class = "abclass.control")
}
