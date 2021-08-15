##' @importFrom stats predict
predict.malc_logistic_path <- function(object, newx = NULL, newy = NULL, ...)
{
    n_slice <- dim(object$coefficients)[3L]
    ## return results for the training set
    if (is.null(newx)) {
        out <- lapply(seq_len(n_slice), function(i) {
                       tmp <- rcpp_predict_cat(object$class_prob[, , i])
                       list(
                           class_prob = object$class_prob[, , i],
                           predicted = tmp,
                           accuracy = mean(tmp == object$category$y)
                       )
                   })
        return(out)
    }
    ## else
    if (! is.matrix(newx)) {
        newx <- as.matrix(newx)
    }
    if (object$intercept) {
        newx <- cbind(1, newx)
    }
    newy <- null2num0(newy)
    lapply(seq_len(n_slice), function(i) {
        tmp <- rcpp_accuracy(newx, newy, object$coefficients[, , i])
        if (is.nan(tmp$accuracy)) tmp$accuracy <- NA_real_
        tmp
    })
}
