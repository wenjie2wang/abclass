##' @importFrom stats predict
predict.abclass_logistic_net <- function(object, newx, newy = NULL, ...)
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
    newy <- if (! is.null(newy)) {
                char_y <- as.character(newy)
                all_y_cat <- unique(c(object$category$label,
                                      sort(unique(char_y))))
                factor_y <- factor(char_y, levels = all_y_cat)
                as.integer(factor_y) - 1L
            } else {
                null2num0(newy)
            }
    out <- rcpp_accuracy(newx, newy, object$coefficients)
    out$predicted <- out$predicted + 1L
    if (is.nan(out$accuracy))
        out$accuracy <- NA_real_
    out
}


predict.abclass_logistic_net_path <- function(object, newx, newy = NULL, ...)
{
    n_slice <- dim(object$coefficients)[3L]
    if (missing(newx)) {
        stop("The 'newx' must be specified.")
    }
    if (! is.matrix(newx)) {
        newx <- as.matrix(newx)
    }
    if (object$intercept) {
        newx <- cbind(1, newx)
    }
    newy <- if (! is.null(newy)) {
                char_y <- as.character(newy)
                all_y_cat <- unique(c(object$category$label,
                                      sort(unique(char_y))))
                factor_y <- factor(char_y, levels = all_y_cat)
                as.integer(factor_y) - 1L
            } else {
                null2num0(newy)
            }
    lapply(seq_len(n_slice), function(i) {
        tmp <- rcpp_accuracy(newx, newy, object$coefficients[, , i])
        if (is.nan(tmp$accuracy)) tmp$accuracy <- NA_real_
        tmp$predicted <- tmp$predicted + 1L
        tmp
    })
}
