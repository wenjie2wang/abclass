##' Prediction by A Trained Angle-Based Classifier
##'
##' Predict class labels or estimate conditional probabilities for the specified
##' new data.
##'
##' @param object An object of class \code{abclass}.
##' @param newx A numeric matrix representing the design matrix for predictions.
##' @param type A character value specifying the desired type of predictions.
##'     The available options are \code{"class"} for predicted labels and
##'     \code{"probability"} for class conditional probability estimates.
##' @param selection A character value specifying how to select a particular set
##'     of coefficient estimates from the solution path for the predictions.
##'     All the
##' @param ... Other arguments not used now.
##'
##' @return A list containing the predictions.
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
    if (! is.matrix(newx)) {
        newx <- as.matrix(newx)
    }
    if (object$intercept) {
        newx <- cbind(1, newx)
    }
    type <- match.arg(type, c("class", "probability"))
    n_slice <- dim(object$coefficients)[3L]
    ## set the selection index
    selection <- match.arg(selection, c("cv_min", "cv_1se", "all"))
    if (! length(object$cross_validation$cv_accuracy) || selection == "all") {
        selection_idx <- seq_len(n_slice)
    } else {
        cv_idx_list <- with(object$cross_validation,
                            select_lambda(cv_accuracy_mean, cv_accuracy_sd))
        selection_idx <- cv_idx_list[[selection]]
    }
    ## determine the internal function to call
    loss_fun <- gsub("-", "_", object$loss$loss, fixed = TRUE)
    predict_prob_fun <- sprintf("rcpp_%s_predict_prob", loss_fun)
    predict_class_fun <- sprintf("rcpp_%s_predict_y", loss_fun)
    arg_list <- switch(
        loss_fun,
        "logistic" = list(),
        "boost" = with(object$loss, list(inner_min = boost_umin)),
        "hinge_boost" = with(object$loss, list(lum_c = lum_c)),
        "lum" = with(object$loss, list(lum_a = lum_a, lum_c = lum_c))
    )
    arg_list$x <- newx
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
