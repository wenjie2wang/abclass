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

##' Prediction by A Trained Angle-Based Classifier
##'
##' Predict class labels or estimate conditional probabilities for the specified
##' new data.
##'
##' @inheritParams coef.abclass
##'
##' @param newx A numeric matrix representing the design matrix for predictions.
##' @param type A character value specifying the desired type of predictions.
##'     The available options are \code{"class"} for predicted labels and
##'     \code{"probability"} for class conditional probability estimates.
##' @param selection An integer vector for the solution indices or a character
##'     value specifying how to select a particular set of coefficient estimates
##'     from the entire solution path for prediction. If the specified
##'     \code{abclass} object contains the cross-validation results, one may set
##'     \code{selection} to \code{"cv_min"} (or \code{"cv_1se"}) for using the
##'     estimates giving the smallest cross-validation error (or the set of
##'     estimates resulted from the largest \emph{lambda} within one standard
##'     error of the smallest cross-validation error) or prediction.  The
##'     prediction for the entire solution path will be returned in a list if
##'     \code{selection = "all"} or no cross-validation results are available in
##'     the specified \code{abclass} object.
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
                            type = c("class", "probability"),
                            selection = c("cv_min", "cv_1se", "all"),
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
    type <- match.arg(type, c("class", "probability"))
    n_slice <- dim(object$coefficients)[3L]
    ## set the selection index
    if (is.na(n_slice) || n_slice == 1L) {
        selection_idx <- 1L
    }
    ## for integer indices
    if (is.numeric(selection)) {
        selection <- as.integer(selection)
        if (any(selection > n_slice)) {
            stop(sprintf("The integer 'selection' must <= %d.", n_slice))
        }
        selection_idx <- selection
    } else {
        selection <- match.arg(selection, c("cv_min", "cv_1se", "all"))
        if (! length(object$cross_validation$cv_accuracy) ||
            selection == "all") {
            selection_idx <- seq_len(n_slice)
        } else {
            cv_idx_list <- object$cross_validation
            selection_idx <- cv_idx_list[[selection]]
        }
    }
    ## determine the internal function to call
    loss_fun <- gsub("-", "_", object$loss$loss, fixed = TRUE)
    predict_prob_fun <- sprintf("r_%s_pred_prob", loss_fun)
    predict_class_fun <- sprintf("r_%s_pred_y", loss_fun)
    if (is_x_sparse) {
        predict_prob_fun <- paste0(predict_prob_fun, "_sp")
        predict_class_fun <- paste0(predict_class_fun, "_sp")
    }
    arg_list <- list(x = newx)
    pred_list <- switch(
        type,
        "class" = {
            lapply(selection_idx, function(i) {
                arg_list$beta <- as.matrix(object$coefficients[, , i])
                tmp <- do.call(predict_class_fun, arg_list)
                z2cat(as.integer(tmp), object$category)
            })
        },
        "probability" = {
            lapply(selection_idx, function(i) {
                arg_list$beta <- as.matrix(object$coefficients[, , i])
                tmp <- do.call(predict_prob_fun, arg_list)
                colnames(tmp) <- object$category$label
                rownames(tmp) <- rownames(newx)
                tmp
            })
        }
    )
    ## return
    if (length(pred_list) == 1L) {
        return(pred_list[[1L]])
    }
    pred_list
}
