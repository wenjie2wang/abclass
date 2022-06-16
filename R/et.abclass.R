##' Tune Angle-Based Classifiers by ET-Lasso
##'
##' Tune the regularization parameter for an angle-based large-margin classifier
##' by ET-Lasso (Yang, et al., 2019).
##'
##' @inheritParams abclass
##'
##' @param nstage A positive integer specifying for the number of stages in the
##'     ET-Lasso procedure.
##'
##' @references
##'
##' Yang, S., Wen, J., Zhan, X., & Kifer, D. (2019). ET-Lasso: A new efficient
##' tuning of lasso-type regularization for high-dimensional data. In
##' /emph{Proceedings of the 25th ACM SIGKDD International Conference on
##' Knowledge Discovery \& Data Mining} (pp. 607--616).
##'
##' @export
et.abclass <- function(x, y,
                       intercept = TRUE,
                       weight = NULL,
                       loss = c("logistic", "boost", "hinge-boost", "lum"),
                       control = list(),
                       nstages = 1,
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
