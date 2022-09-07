##' @importFrom stats BIC
##' @export
BIC.logistic_suplasso <- function(object, ...)
{
    nll <- attr(object$coefficients, "negLogL")
    k <- apply(object$coefficients, 3L, function(a) {
        sum(abs(a) > .Machine$double.eps)
    })
    nobs <- length(object$category$y)
    log(nobs) * k + 2 * nll * nobs
}

##' @export
BIC.logistic_supscad <- BIC.logistic_suplasso
