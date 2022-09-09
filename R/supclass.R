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

##' Multi-Category Classifiers with Sup-Norm Regularization
##'
##' Experimental implementations of multi-category classifiers with sup-norm
##' penalties proposed by Zhang, et al. (2008) and Li & Zhang (2021).
##'
##' For the multinomial logistic model or the proximal SVM model, this function
##' utilizes the function \code{quadprog::solve.QP()} to solve the equivalent
##' quadratic problem; For the multi-class SVM, this function utilizes GNU GLPK
##' to solve the equivalent linear programming problem via the package {Rglpk}.
##' It is recommended to use a recent version of {GLPK}.
##'
##' @inheritParams abclass
##'
##' @param model A charactor vector specifying the classification model.  The
##'     available options are \code{"logistic"} for multi-nomial logistic
##'     regression model, \code{"psvm"} for proximal support vector machine
##'     (PSVM), \code{"svm"} for multi-category support vector machine.
##' @param penalty A charactor vector specifying the penalty function for the
##'     sup-norms.  The available options are \code{"lasso"} for sup-norm
##'     regularization proposed by Zhang et al. (2008) and \code{"scad"} for
##'     supSCAD regularization proposed by Li & Zhang (2021).
##' @param start A numeric matrix representing the starting values for the
##'     quadratic approximation procedure behind the scene.
##' @param control A list with named elements.
##' @param ... Optional control parameters passed to the
##'     \code{supclass.control()}.
##'
##' @references
##'
##' Zhang, H. H., Liu, Y., Wu, Y., & Zhu, J. (2008). Variable selection for the
##' multicategory SVM via adaptive sup-norm regularization. \emph{Electronic
##' Journal of Statistics}, 2, 149--167.
##'
##' Li, N., & Zhang, H. H. (2021). Sparse learning with non-convex penalty in
##' multi-classification. \emph{Journal of Data Science}, 19(1), 56--74.
##'
##' @example inst/examples/ex-supclass.R
##'
##' @export
supclass <- function(x, y,
                     model = c("logistic", "psvm", "svm"),
                     penalty = c("lasso", "scad"),
                     start = NULL,
                     control = list(),
                     ...)
{
    ## select models
    model <- match.arg(as.character(model),
                       choices = c("logistic", "psvm", "svm"))
    if (model %in% c("logistic", "psvm")) {
        suggest_pkg("qpmadr")
    } else {
        suggest_pkg("Rglpk")
    }
    ## select panalty
    penalty <- match.arg(as.character(penalty),
                         choices = c("lasso", "scad"))
    ## pre-process x and y
    if (! is.matrix(x)) {
        x <- as.matrix(x)
    }
    cat_y <- cat2z(y, zero_based = FALSE)
    K <- max(cat_y$y)
    p <- ncol(x)
    pp <- p + 1L
    ## controls
    dot_list <- list(...)
    control <- do.call(supclass.control, modify_list(control, dot_list))
    ## standardize x if needed
    x_center <- x_scale <- NULL
    if (control$standardize) {
        x_center <- colMeans(x)
        x_scale <- apply(x, 2L, function(a) {
            sqrt(mean((a - mean(a)) ^ 2))
        })
        for (j in seq_len(p)) {
            if (x_scale[j] > 0) {
                x[, j] <- (x[, j] - x_center[j]) / x_scale[j]
            } else {
                x[, j] <- 0
                x_scale[j] <- - 1 # for ease of scaling back later
            }
        }
    }
    ## adaptive weights for lasso
    if (penalty == "lasso") {
        if (is.null(control$adaptive_weight)) {
            control$adaptive_weight <- rep(1, p)
            control$is_adaptive_mat <- FALSE
        } else {
            is_adaptive_mat <- is.matrix(control$adaptive_weight)
            if (is_adaptive_mat &&
                any(dim(control$adaptive_weight) != c(p, k))) {
                stop(sprintf(
                    "The adaptive weight matrix must be %d by %d.",
                    p, k), call. = FALSE)
            }
            if (! is_adaptive_mat && length(control$adaptive_weight) != p) {
                stop(sprintf(
                    "The length of the adaptive weight vector must be %d.", p),
                    call. = FALSE)
            }
            if (any(control$adaptive_weight < 0)) {
                stop("The adaptive weights must be nonnegative.")
            }
            control$is_adaptive_mat <- is_adaptive_mat
        }
    }
    ## check start
    if (is.null(start)) {
        ## TODO apply ridge estimates instead of zeros
        start <- matrix(0, nrow = pp, ncol = K)
    } else {
        if (! is.matrix(start)) {
            start <- as.matrix(start)
        }
        if (nrow(start) != pp || ncol(start) != K) {
            stop(sprintf(
                "The starting value should be a %d by %d matrix", pp, K
            ), call. = FALSE)
        }
    }
    ## call the corresponding function
    beta <- switch(
        model,
        "logistic" = supclass_mlog(x, cat_y$y, penalty, start, control),
        "psvm" = supclass_mpsvm(x, cat_y$y, penalty, start, control),
        "svm" = supclass_msvm(x, cat_y$y, penalty, start, control)
    )
    ## impose shrinkage for every slice
    for (l in seq_along(control$lambda)) {
        beta[rowSupnorms(beta[, , l]) < control$shrinkage, , l] <- 0
    }
    ## beta[abs(beta) < control$shrinkage] <- 0
    ## scale the estimate back for the original scale of x
    if (control$standardize) {
        for (l in seq_along(control$lambda)) {
            for (k in seq_len(K)) {
                ## for intercept
                beta[1, k, l] <- beta[1, k, l] -
                    sum(beta[- 1, k, l] * x_center / x_scale)
                beta[- 1, k, l] <- beta[- 1, k, l] / x_scale
            }
        }
    }
    ## prepare return
    regus <- if (penalty == "lasso") {
                 c("lambda", "adaptive_weight")
             } else {
                 c("lambda", "scad_a")
             }
    ctrls <- c("maxit", "epsilon", "shrinkage", "warm_start",
               "standardize", "verbose")
    structure(list(
        coefficients = beta,
        category = cat_y,
        model = model,
        regularization = control[regus],
        start = start,
        control = control[ctrls]
    ), class = c(sprintf("%s_sup%s", model, penalty), "supclass"))
}


##' @rdname supclass
##'
##' @inheritParams abclass.control
##'
##' @param lambda A numeric vector specifying the tuning parameter
##'     \emph{lambda}.  The default value is \code{0.1}.  Users should tune this
##'     parameter for a better model fit.  The specified lambda will be sorted
##'     in decreasing order internally and only the unique values will be kept.
##' @param adaptive_weight A numeric vector or matrix representing the adaptive
##'     penalty weights.  The default value is \code{NULL} for equal weights.
##'     Zhang, et al. (2008) proposed two ways to employ the adaptive weights.
##'     The first approach applies the weights to the sup-norm of coefficient
##'     estimates, while the second approach applies element-wise multiplication
##'     to the weights and coefficient estimates inside the sup-norms.  The
##'     first or second approach will be applied if a numeric vector or matrix
##'     is specified, respectively.  The adaptive weights are supported for
##'     lasso penalty only.
##' @param scad_a A positive number specifying the tuning parameter \emph{a} in
##'     the SCAD penalty.
##' @param maxit A positive integer specifying the maximum number of iteration.
##'     The default value is \code{50} as suggested in Li & Zhang (2021).
##' @param shrinkage A nonnegative tolerance to shrink estimates with sup-norm
##'     close enough to zero (within the specified tolerance) to zeros.  The
##'     default value is \code{1e-4}.  ## @param ridge_lambda The tuning
##'     parameter lambda of the ridge penalty used to ## set the (first set of)
##'     starting values.
##' @param warm_start A logical value indicating if the estimates from last
##'     lambda should be used as the starting values for the next lambda.  If
##'     \code{FALSE}, the user-specified starting values will be used instead.
##' @param standardize A logical value indicating if a standardization procedure
##'     should be performed so that each column of the design matrix has mean
##'     zero and standardization
##'
##' @export
supclass.control <- function(lambda = 0.1,
                             adaptive_weight = NULL,
                             scad_a = 3.7,
                             maxit = 50,
                             epsilon = 1e-4,
                             shrinkage = 1e-4,
                             ## ridge_lambda = 1e-4,
                             warm_start = TRUE,
                             standardize = TRUE,
                             verbose = 0L,
                             ...)
{
    structure(list(
        lambda = sort(unique(lambda), decreasing = TRUE),
        adaptive_weight = adaptive_weight,
        scad_a = scad_a,
        maxit = maxit,
        epsilon = epsilon,
        shrinkage = shrinkage,
        warm_start = warm_start,
        standardize = standardize,
        verbose = verbose
    ), class = "supclass.control")
}


### internals

## scadp <- function(a, lambda, eta)
## {
##     aeta <- abs(eta)
##     lambda * aeta * as.numeric(aeta <= lambda) -
##         as.numeric(aeta > lambda & aeta <= a * lambda) *
##         (aeta ^ 2 - 2 * a * lambda * aeta + lambda ^ 2) / (2 * a - 2) +
##         0.5 * (a + 1) * lambda ^ 2 * as.numeric(aeta > a * lambda)
## }

## first derivatives of SCAD
scaddp <- function(a, lambda, eta)
{
    aeta <- abs(eta)
    lambda * as.numeric(aeta <= lambda) +
        as.numeric(aeta > lambda & aeta <= a * lambda) *
        (a * lambda - aeta) / (a - 1)
}

## likelihood function for the multinomial logistic model
## along with gradient and hessian
deriv_mlog <- function(x, y, beta)
{
    n <- nrow(x)
    p <- nrow(beta)
    K <- ncol(beta)
    pK <- p * K
    exp_xb <- exp(x %*% beta)
    sum_exp_xb <- rowSums(exp_xb)
    prob <- exp_xb / sum_exp_xb
    lnp <- log(prob[cbind(seq_along(y), y)])
    ## nlnl <- - mean(lnp) # negative log-likelihood
    ## gradient
    y_mat <- matrix(0L, nrow = n, ncol = K)
    y_mat[cbind(seq_len(n), y)] <- 1L
    grad_vec <- as.numeric(t(x) %*% (prob - y_mat))
    ## hessian
    hess_mat <- matrix(0, nrow = pK, ncol = pK)
    for (i in seq_len(n)) {
        xxt <- tcrossprod(x[i, ])
        p_i <- prob[i, ]
        p_i <- diag(p_i) - tcrossprod(p_i)
        hess_mat <- hess_mat + kronecker(p_i, xxt)
    }
    ## return
    list(
        ## prob = prob,
        ## neglogl = nlnl,
        grad = grad_vec,
        hess = hess_mat
    )
}

## negative log-likelihood
nll_mlog <- function(x, y, beta) {
    exp_xb <- exp(x %*% beta)
    sum_exp_xb <- rowSums(exp_xb)
    prob <- exp_xb / sum_exp_xb
    lnp <- log(prob[cbind(seq_along(y), y)])
    - mean(lnp)
}

## multinomial logistic model with sup-norm penalties
supclass_mlog <- function(x, y, penalty, start, control)
{
    K <- max(y)
    n <- nrow(x)
    p <- ncol(x)
    pp <- p + 1L
    ppK <- pp * K
    ## convert beta estimates in vector to matrix
    get_beta <- function(theta) {
        matrix(theta[seq_len(ppK)], ncol = K)
    }
    get_eta <- function(theta) {
        theta[- seq_len(ppK)]
    }
    x <- cbind(1, x)
    df <- ppK + p
    ## helper to check variable names
    ## .get_var_names <- function(x) {
    ##     gen_names <- function(na, nb, a_zero_based = FALSE,
    ##                           b_zero_based = FALSE) {
    ##         do.call(
    ##             function(a, b) sprintf("%d_%d", a, b),
    ##             as.list(expand.grid(
    ##                 a = seq_len(na) - as.integer(a_zero_based),
    ##                 b = seq_len(nb) - as.integer(b_zero_based)
    ##             ))
    ##         )
    ##     }
    ##     var_names <- c(
    ##         paste0("beta_", gen_names(pp, K, TRUE)),
    ##         paste0("eta_", seq_len(p))
    ##     )
    ##     idx <- x != 0
    ##     setNames(x[idx], var_names[idx])
    ## }
    ## get_var_names <- function(x) {
    ##     apply(as.matrix(x), 2, .get_var_names, simplify = FALSE)
    ## }
    ## TODO avoid t(Amat) by generating Amat in a different way
    ## prepare the matrix for the equality constraints
    A0 <- matrix(0, nrow = df, ncol = pp)
    for (i in seq_len(pp)) {
        A0[seq.int(i, by = pp, length.out = K), i] <- 1
    }
    ## prepare the matrix for the inequality constraints
    Aineq <- if (isTRUE(control$is_adaptive_mat)) {
                 lapply(seq_len(K), function(k) {
                     A1 <- matrix(0, nrow = df, ncol = p)
                     idx_mat <- cbind((k - 1) * pp + 1 + seq_len(p), seq_len(p))
                     A1[idx_mat] <- - control$adaptive_weight[, k]
                     A1[cbind(seq_len(p) + ppK, seq_len(p))] <- 1
                     cbind(A1, abs(A1))
                 })
             } else {
                 ## abs(beta_k) <= eta_k
                 lapply(seq_len(K), function(k) {
                     A1 <- matrix(0, nrow = df, ncol = p)
                     idx_mat <- cbind((k - 1) * pp + 1 + seq_len(p), seq_len(p))
                     A1[idx_mat] <- - 1
                     A1[cbind(seq_len(p) + ppK, seq_len(p))] <- 1
                     cbind(A1, abs(A1))
                 })
             }
    Aineq <- do.call(cbind, Aineq)
    Amat <- cbind(A0, Aineq)
    b0vec <- rep(0, pp + 2 * p * K)
    sc <- sqrt(.Machine$double.eps)
    ## initialize
    beta_array <- array(NA, dim = c(pp, K, length(control$lambda)))
    nll_vec <- rep(NA, length(control$lambda))
    ## for a sequence of lambda's
    for (l in seq_along(control$lambda)) {
        outer_beta0 <- if (l > 1 && control$warm_start) {
                           beta1
                       } else {
                           start
                       }
        eta <- apply(abs(outer_beta0[- 1L, ]), 1, max)
        inner_iter <- outer_iter <- 0
        ## main loop for one single lambda
        while (outer_iter < control$maxit) {
            outer_iter <- outer_iter + 1L
            lik_res <- deriv_mlog(x, y, outer_beta0)
            grad_vec <- lik_res$grad / n
            hess_mat <- lik_res$hess / n
            Dmat <- matrix(0, nrow = df, ncol = df)
            Dmat[seq_len(ppK), seq_len(ppK)] <- hess_mat
            diag(Dmat) <- diag(Dmat) + sc
            dvec0 <- grad_vec - hess_mat %*% as.numeric(outer_beta0)
            inner_beta0 <- outer_beta0
            if (penalty == "lasso") {
                dp <- if (control$is_adaptive_mat) {
                          rep(control$lambda[l], p)
                      } else {
                          control$lambda[l] * control$adaptive_weight
                      }
                dvec <- c(dvec0, dp)
                qres <- tryCatch({
                    ## quadprog::solve.QP(Dmat = Dmat,
                    ##                    dvec = - dvec,
                    ##                    Amat = Amat,
                    ##                    bvec = b0vec,
                    ##                    meq = pp)
                    qpmadr::solveqp(
                                H = Dmat,
                                h = dvec,
                                A = t(Amat),
                                Alb = b0vec,
                                Aub = c(rep(0, pp), rep(Inf, 2 * p * K))
                            )
                }, error = function(e) e)
                beta1 <- if (inherits(qres, "error")) {
                             warning(qres,
                                     "\nRevert to the solution from last step.")
                             outer_beta0
                         } else {
                             get_beta(qres$solution)
                         }
                if (anyNA(beta1)) {
                    warning("Found NA in the beta estimates.",
                            "The specified lambda was probably too small.",
                            "\nRevert to the solution from last step.")
                    beta1 <- outer_beta0
                }
            } else {
                while (inner_iter < control$maxit) {
                    inner_iter <- inner_iter + 1
                    dp <- scaddp(control$scad_a, control$lambda[l], eta)
                    dvec <- c(dvec0, dp)
                    qres <- tryCatch({
                        ## quadprog::solve.QP(Dmat = Dmat,
                        ##                    dvec = - dvec,
                        ##                    Amat = Amat,
                        ##                    bvec = b0vec,
                        ##                    meq = pp)
                        qpmadr::solveqp(
                                    H = Dmat,
                                    h = dvec,
                                    A = t(Amat),
                                    Alb = b0vec,
                                    Aub = c(rep(0, pp), rep(Inf, 2 * p * K))
                                )
                    }, error = function(e) e)
                    if (inherits(qres, "error")) {
                        warning(
                            qres,
                            "\nRevert to the soltion from last step."
                        )
                        beta1 <- inner_beta0
                    } else {
                        beta1 <- get_beta(qres$solution)
                        eta <- get_eta(qres$solution)
                    }
                    if (anyNA(beta1)) {
                        warning("Found NA in the beta estimates.",
                                "The specified lambda was probably too small.",
                                "\nRevert to the solution from last step.")
                        beta1 <- inner_beta0
                    }
                    inner_diff <- rowL2sums(beta1 - inner_beta0)
                    if (inner_diff < control$epsilon) {
                        break
                    }
                    inner_beta0 <- beta1
                }
            }
            outer_diff <- rowL2sums(beta1 - outer_beta0)
            if (outer_diff < control$epsilon) {
                break
            }
            outer_beta0 <- beta1
        }
        beta_array[, , l] <- beta1
        ## for computing BIC for logistic model later
        nll_vec[l] <- nll_mlog(x, y, beta1)
    }
    ## return
    attr(beta_array, "negLogL") <- nll_vec
    beta_array
}

## mpsvm with sup-norm penalties
supclass_mpsvm <- function(x, y, penalty, start, control)
{
    K <- max(y)
    n <- nrow(x)
    p <- ncol(x)
    pp <- p + 1L
    ppK <- pp * K
    ## convert beta estimates in vector to matrix
    get_beta <- function(theta) {
        matrix(theta[seq_len(ppK)], ncol = K)
    }
    get_eta <- function(theta) {
        theta[- seq_len(ppK)]
    }
    x <- cbind(1, x)
    df <- ppK + p
    ## prepare the matrix for the equality constraints
    ## sum of beta associated with one variable = 0
    A0 <- matrix(0, nrow = df, ncol = pp)
    for (i in seq_len(pp)) {
        A0[seq.int(i, by = pp, length.out = K), i] <- 1
    }
    ## prepare the matrix for the inequality constraints
    Aineq <- if (isTRUE(control$is_adaptive_mat)) {
                 lapply(seq_len(K), function(k) {
                     A1 <- matrix(0, nrow = df, ncol = p)
                     idx_mat <- cbind((k - 1) * pp + 1 + seq_len(p), seq_len(p))
                     A1[idx_mat] <- - control$adaptive_weight[, k]
                     A1[cbind(seq_len(p) + ppK, seq_len(p))] <- 1
                     cbind(A1, abs(A1))
                 })
             } else {
                 ## abs(beta_k) <= eta_k
                 lapply(seq_len(K), function(k) {
                     A1 <- matrix(0, nrow = df, ncol = p)
                     idx_mat <- cbind((k - 1) * pp + 1 + seq_len(p), seq_len(p))
                     A1[idx_mat] <- - 1
                     A1[cbind(seq_len(p) + ppK, seq_len(p))] <- 1
                     cbind(A1, abs(A1))
                 })
             }
    Aineq <- do.call(cbind, Aineq)
    Amat <- cbind(A0, Aineq)
    b0vec <- rep(0, pp + 2 * p * K)
    ## prepare the vector for the objective function
    delta <- matrix(1, nrow = n, ncol = K) / n
    delta[cbind(seq_len(n), y)] <- 0
    dvec0 <- 2 * as.numeric(crossprod(x, delta))
    Dmat <- matrix(0, nrow = df, ncol = df)
    for (k in seq_len(K)) {
        idx <- seq.int(1 + (k - 1) * pp, k * pp)
        Dmat[idx, idx] <- crossprod(delta[, k] * x, x)
    }
    sc <- sqrt(.Machine$double.eps)
    diag(Dmat) <- diag(Dmat) + sc
    ## initialize
    beta_array <- array(NA, dim = c(pp, K, length(control$lambda)))
    ## for a sequence of lambda's
    for (l in seq_along(control$lambda)) {
        beta0 <- if (l > 1 && control$warm_start) {
                     beta1
                 } else {
                     start
                 }
        if (penalty == "lasso") {
            dp <- if (control$is_adaptive_mat) {
                      rep(control$lambda[l], p)
                  } else {
                      control$lambda[l] * control$adaptive_weight
                  }
            dvec <- c(dvec0, dp)
            qres <- tryCatch({
                ## quadprog::solve.QP(Dmat = Dmat,
                ##                    dvec = - dvec,
                ##                    Amat = Amat,
                ##                    bvec = b0vec,
                ##                    meq = pp)
                qpmadr::solveqp(
                            H = Dmat,
                            h = dvec,
                            A = t(Amat),
                            Alb = b0vec,
                            Aub = c(rep(0, pp), rep(Inf, 2 * p * K))
                        )
            }, error = function(e) e)
            beta1 <- if (inherits(qres, "error")) {
                         warning(qres,
                                 "\nRevert to the solution from last step.")
                         beta0
                     } else {
                         get_beta(qres$solution)
                     }
            if (anyNA(beta1)) {
                warning("Found NA in the beta estimates.",
                        "The specified lambda was probably too small.",
                        "\nRevert to the solution from last step.")
                beta1 <- beta0
            }
        } else {
            ## main loop for one single lambda
            iter <- 0L
            eta <- apply(abs(beta0[- 1L, ]), 1, max)
            while (iter < control$maxit) {
                iter <- iter + 1L
                dp <- scaddp(control$scad_a, control$lambda[l], eta)
                dvec <- c(dvec0, dp)
                qres <- tryCatch({
                    ## quadprog::solve.QP(Dmat = Dmat,
                    ##                    dvec = - dvec,
                    ##                    Amat = Amat,
                    ##                    bvec = b0vec,
                    ##                    meq = pp)
                    qpmadr::solveqp(
                                H = Dmat,
                                h = dvec,
                                A = t(Amat),
                                Alb = b0vec,
                                Aub = c(rep(0, pp), rep(Inf, 2 * p * K))
                            )
                }, error = function(e) e)
                if (inherits(qres, "error")) {
                    warning(qres, "\nRevert to the solution from last step.")
                    beta1 <- beta0
                } else {
                    beta1 <- get_beta(qres$solution)
                    eta <- get_eta(qres$solution)
                }
                if (anyNA(beta1)) {
                    warning("Found NA in the beta estimates.",
                            "The specified lambda was probably too small.",
                            "\nRevert to the solution from last step.")
                    beta1 <- beta0
                }
                tol <- rowL2sums(beta1 - beta0)
                if (tol < control$epsilon) {
                    break
                }
                beta0 <- beta1
            }
        }
        beta_array[, , l] <- beta1
    }
    ## return
    beta_array
}

## msvm with sup-norm penalties
supclass_msvm <- function(x, y, penalty, start, control)
{
    K <- max(y)
    n <- nrow(x)
    p <- ncol(x)
    pp <- p + 1L
    ppK <- pp * K
    nK <- n * K
    ## convert beta estimates in vector to matrix
    get_beta <- function(theta) {
        matrix(theta[seq_len(ppK)], byrow = FALSE, ncol = K)
    }
    ## get_xi <- function(theta) {
    ##     matrix(theta[seq.int(ppK + 1, by = 1, length.out = nK)],
    ##            nrow = n, ncol = K)
    ## }
    get_eta <- function(theta) {
        theta[seq.int(ppK + nK + 1, by = 1, length.out = p)]
    }
    ## helper to check variable names
    ## .get_var_names <- function(x) {
    ##     gen_names <- function(na, nb, a_zero_based = FALSE,
    ##                           b_zero_based = FALSE) {
    ##         do.call(
    ##             function(a, b) sprintf("%d_%d", a, b),
    ##             as.list(expand.grid(
    ##                 a = seq_len(na) - as.integer(a_zero_based),
    ##                 b = seq_len(nb) - as.integer(b_zero_based)
    ##             ))
    ##         )
    ##     }
    ##     var_names <- c(
    ##         paste0("beta_", gen_names(pp, K, TRUE)),
    ##         paste0("xi_", gen_names(n, K)),
    ##         paste0("eta_", seq_len(p))
    ##     )
    ##     idx <- x != 0
    ##     setNames(x[idx], var_names[idx])
    ## }
    ## get_var_names <- function(x) {
    ##     apply(as.matrix(x), 1, .get_var_names, simplify = FALSE)
    ## }
    ## data
    x <- cbind(1, x)
    df <- ppK + nK + p
    ## prepare the matrix for the equality constraints
    ## sum of beta associated with one variable = 0
    A0 <- matrix(0, nrow = pp, ncol = df)
    for (i in seq_len(pp)) {
        A0[i, seq.int(i, by = pp, length.out = K)] <- 1
    }
    ## prepare the matrix for the inequality constraints
    ## xi_i,k - x_i^T beta_k >= 1
    A1s <- lapply(seq_len(K), function(k) {
        A1 <- matrix(0, nrow = n, ncol = df)
        A1[, seq.int(1 + (k - 1) * pp, k * pp)] <- - x
        A1[cbind(seq_len(n), seq.int(ppK + 1 + (k - 1) * n, ppK + n * k))] <- 1
        A1
    })
    A1s <- do.call(rbind, A1s)
    Aineq <- if (isTRUE(control$is_adaptive_mat)) {
                 ## abs(beta_k * adaptive_weight_k) <= eta_k
                 lapply(seq_len(p), function(j) {
                     A3 <- matrix(0, nrow = K, ncol = df)
                     idx_mat <- cbind(seq_len(K),
                                      seq.int(1 + j, by = pp, length.out = K))
                     A3[idx_mat] <- - control$adaptive_weight[j, ]
                     A3[, ppK + nK + j] <- 1
                     rbind(A3, abs(A3))
                 })
             } else {
                 ## abs(beta_k) <= eta_k
                 lapply(seq_len(p), function(j) {
                     A3 <- matrix(0, nrow = K, ncol = df)
                     idx_mat <- cbind(seq_len(K),
                                      seq.int(1 + j, by = pp, length.out = K))
                     A3[idx_mat] <- - 1
                     A3[, ppK + nK + j] <- 1
                     rbind(A3, abs(A3))
                 })
             }
    Aineq <- do.call(rbind, Aineq)
    Amat <- rbind(A0, A1s, Aineq)
    b0vec <- rep(0, pp + nK + 2 * p * K)
    b0vec[seq.int(pp + 1, pp + nK)] <- 1
    ## prepare the vector for the objective function
    delta <- matrix(1, nrow = n, ncol = K)
    delta[cbind(seq_len(n), y)] <- 0
    delta <- as.numeric(delta) / n
    const_dir <- c(rep("==", pp), rep(">=", nK + 2 * p * K))
    ## initialize
    beta_array <- array(NA, dim = c(pp, K, length(control$lambda)))
    ## for a sequence of lambda's
    for (l in seq_along(control$lambda)) {
        beta0 <- if (l > 1 && control$warm_start) {
                     beta1
                 } else {
                     start
                 }
        if (penalty == "lasso") {
            dp <- if (control$is_adaptive_mat) {
                      rep(control$lambda[l], p)
                  } else {
                      control$lambda[l] * control$adaptive_weight
                  }
            objective_in <- c(rep(0, ppK), delta, dp)
            lres <- tryCatch({
                Rglpk::Rglpk_solve_LP(obj = objective_in,
                                      mat = Amat,
                                      dir = const_dir,
                                      rhs = b0vec,
                                      bounds = list(
                                          lower = list(
                                              ind = seq_len(ppK),
                                              val = - rep(Inf, ppK)
                                          )
                                      ),
                                      control = list(
                                          verbose = control$verbose
                                          ## presolve = TRUE
                                      ))
                ## Rsymphony::Rsymphony_solve_LP(
                ##                obj = objective_in,
                ##                mat = Amat,
                ##                dir = const_dir,
                ##                rhs = b0vec,
                ##                bounds = list(
                ##                    lower = list(
                ##                        ind = seq_len(ppK),
                ##                        val = - rep(Inf, ppK)
                ##                    )
                ##                ),
                ##                verbosity = - 2
                ##                ## write_mps = TRUE,
                ##                ## write_lp = TRUE
                ##            )
            }, error = function(e) e)
            beta1 <- if (inherits(lres, "error")) {
                         warning(lres,
                                 "\nRevert to the solution from last step.")
                         beta0
                     } else {
                         get_beta(lres$solution)
                     }
        } else {
            ## main loop for one single lambda
            iter <- 0L
            eta <- apply(abs(beta0[- 1L, ]), 1, max)
            while (iter < control$maxit) {
                iter <- iter + 1L
                dp <- scaddp(control$scad_a, control$lambda[l], eta)
                objective_in <- c(rep(0, ppK), delta, dp)
                lres <- tryCatch(
                    Rglpk::Rglpk_solve_LP(obj = objective_in,
                                          mat = Amat,
                                          dir = const_dir,
                                          rhs = b0vec,
                                          bounds = list(
                                              lower = list(
                                                  ind = seq_len(ppK),
                                                  val = - rep(Inf, ppK)
                                              )
                                          ),
                                          control = list(
                                              verbose = control$verbose,
                                              presolver = TRUE
                                          )),
                    error = function(e) e
                )
                if (inherits(lres, "error")) {
                    warning(lres, "\nRevert to the solution from last step.")
                    beta1 <- beta0
                } else {
                    eta <- get_eta(lres$solution)
                    beta1 <- get_beta(lres$solution)
                }
                tol <- rowL2sums(beta1 - beta0)
                if (tol < control$epsilon) {
                    break
                }
                beta0 <- beta1
            }
        }
        beta_array[, , l] <- beta1
    }
    ## return
    beta_array
}
