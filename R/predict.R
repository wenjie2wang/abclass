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

##' Prediction by A Trained Angle-Based Classifier
##'
##' Predict class labels or estimate conditional probabilities for the specified
##' new data.
##'
##' @inheritParams coef.abclass
##'
##' @param newx A numeric matrix representing the design matrix for predictions.
##' @param type A character value specifying the desired type of predictions.
##'     The available options are \code{"class"} for predicted labels,
##'     \code{"probability"} for class conditional probability estimates, and
##'     \code{"link"} for decision functions.
##' @param selection An integer vector for the solution indices or a character
##'     value specifying how to select a particular set of coefficient estimates
##'     from the entire solution path for prediction. If the specified
##'     \code{object} contains the cross-validation results, one may set
##'     \code{selection} to \code{"cv_min"} (or \code{"cv_1se"}) for using the
##'     estimates giving the smallest cross-validation error (or the set of
##'     estimates resulted from the largest \emph{lambda} within one standard
##'     error of the smallest cross-validation error) or prediction.  The
##'     prediction for the entire solution path will be returned in a list if
##'     \code{selection = "all"} or no cross-validation results are available in
##'     the specified \code{object}.
##' @param newoffset An optional numeric matrix for the offsets.
##'
##' @return A vector representing the predictions or a list containing the
##'     predictions for each set of estimates along the solution path.
##'
##' @examples
##' ## see examples of `abclass()`.
##'
##' @importFrom stats predict
##' @export
predict.abclass <- function(object,
                            newx,
                            type = c("class", "probability", "link"),
                            selection = c("cv_1se", "cv_min", "all"),
                            newoffset = NULL,
                            ...)
{
    if (missing(newx)) {
        stop("The 'newx' must be specified.")
    }
    is_x_sparse <- FALSE
    if (inherits(newx, "sparseMatrix")) {
        is_x_sparse <- TRUE
    } else if (! is.matrix(newx)) {
        newx <- as.matrix(newx)
    }
    type <- match.arg(type, c("class", "probability", "link"))
    res_coef <- coef(object, selection = selection)
    loss_id <- .id_loss(object$specs$loss)
    loss_params <- object$control[c("boost_umin", "lum_a", "lum_c")]
    predict_prob_fun <- "rcpp_pred_prob"
    predict_class_fun <- "rcpp_pred_y"
    predict_link_fun <- "rcpp_pred_link"
    if (is_x_sparse) {
        predict_prob_fun <- paste0(predict_prob_fun, "_sp")
        predict_class_fun <- paste0(predict_class_fun, "_sp")
        predict_link_fun <- paste0(predict_link_fun, "_sp")
    }
    if (! is.null(newoffset)) {
        if (! is.matrix(newoffset)) {
            newoffset <- as.matrix(newoffset)
        }
        if (nrow(newoffset) != nrow(newx) ||
            ncol(newoffset) != object$category$k - 1) {
            stop(sprintf("The 'newoffset' must be a %d by %d matrix.",
                         nrow(newx), object$category$k))
        }
    }
    arg_list <- list(x = newx, loss_id = loss_id, offset = null2mat0(newoffset))
    if (type == "probability") {
        arg_list$loss_params <- loss_params
    } else if (type == "link") {
        arg_list$loss_id <- NULL
    }
    if (is.matrix(res_coef) || is.vector(res_coef)) {
        arg_list$beta <- as.matrix(res_coef)
        out <- switch(
            type,
            "class" = {
                tmp <- do.call(predict_class_fun, arg_list)
                tmp <- z2cat(as.integer(tmp), object$category)
                ## names(tmp) <- rownames(newx)
                tmp
            },
            "probability" = {
                tmp <- do.call(predict_prob_fun, arg_list)
                colnames(tmp) <- object$category$label
                ## rownames(tmp) <- rownames(newx)
                tmp
            },
            "link" = {
                drop(do.call(predict_link_fun, arg_list))
            })
        return(out)
    }
    ## else
    nslice <- dim(res_coef)[3L]
    ## return
    switch(
        type,
        "class" = {
            lapply(seq_len(nslice), function(i) {
                arg_list$beta <- as.matrix(res_coef[, , i])
                tmp <- do.call(predict_class_fun, arg_list)
                tmp <- z2cat(as.integer(tmp), object$category)
                ## names(tmp) <- rownames(newx)
                tmp
            })
        },
        "probability" = {
            lapply(seq_len(nslice), function(i) {
                arg_list$beta <- as.matrix(res_coef[, , i])
                tmp <- do.call(predict_prob_fun, arg_list)
                colnames(tmp) <- object$category$label
                ## rownames(tmp) <- rownames(newx)
                tmp
            })
        },
        "link" = {
            lapply(seq_len(nslice), function(i) {
                arg_list$beta <- as.matrix(res_coef[, , i])
                drop(do.call(predict_link_fun, arg_list))
            })
        }
    )
}


##' Predictions from A Trained Sup-Norm Classifier
##'
##' Predict class labels or estimate conditional probabilities for the specified
##' new data.
##'
##' @inheritParams predict.abclass
##'
##' @return A vector representing the predictions or a list containing the
##'     predictions for each set of estimates.
##'
##' @examples
##' ## see examples of `supclass()`.
##'
##' @importFrom stats predict
##' @export
predict.supclass <- function(object,
                             newx,
                             type = c("class", "probability", "link"),
                             selection = c("cv_1se", "cv_min", "all"),
                             ...)
{
    type <- match.arg(type, choices = c("class", "probability", "link"))
    if (object$model %in% c("psvm", "svm") && type == "probability") {
        stop(sprintf("Probability estimates are not available for '%s' model.",
                     object$model))
    }
    if (missing(newx)) {
        stop("The 'newx' must be specified.")
    }
    if (! is.matrix(newx)) {
        newx <- as.matrix(newx)
    }
    newx <- cbind(1, newx)
    ## get coefficient estimates
    res_coef <- coef(object, selection = selection)
    if (is.matrix(res_coef)) {
        xbeta <- newx %*% res_coef
        out <- switch(
            type,
            "class" = {
                tmp <- apply(xbeta, 1L, which.max)
                z2cat(as.integer(tmp), object$category, zero_based = FALSE)
            },
            "probability" = {
                exp_xbeta <- exp(xbeta)
                prob <- exp_xbeta / rowSums(exp_xbeta)
                colnames(prob) <- object$category$label
                drop(prob)
            },
            "link" = {
                drop(xbeta)
            })
        return(out)
    }
    ## else
    nslice <- dim(res_coef)[3L]
    ## return
    switch(
        type,
        "class" = {
            lapply(seq_len(nslice), function(i) {
                beta <- as.matrix(res_coef[, , i])
                xbeta <- newx %*% beta
                tmp <- apply(xbeta, 1L, which.max)
                z2cat(as.integer(tmp), object$category, zero_based = FALSE)
            })
        },
        "probability" = {
            lapply(seq_len(nslice), function(i) {
                beta <- as.matrix(res_coef[, , i])
                exp_xbeta <- exp(newx %*% beta)
                prob <- exp_xbeta / rowSums(exp_xbeta)
                colnames(prob) <- object$category$label
                drop(prob)
            })
        },
        "link" = {
            lapply(seq_len(nslice), function(i) {
                beta <- as.matrix(res_coef[, , i])
                drop(newx %*% beta)
            })
        }
    )
}
