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

##' @importFrom graphics axis lines matplot text
##' @export
plot.abclass_path <- function(x, y, ...)
{
    log_lambda <- log(x$regularization$lambda)
    beta_l2norm <- apply(x$coefficients, c(3, 1), function(a) {
        sqrt(sum(a ^ 2))
    })
    inter <- as.integer(x$specs$intercept)
    dot_list <- list(...)
    default_ctrl <- list(
        lty = 1,
        type = "l",
        xlab = expression(log(lambda)),
        ylab = expression(paste(L[2], "(", beta, ")")),
        ylim = c(0, max(beta_l2norm))
    )
    ctrl <- modify_list(default_ctrl, dot_list)
    ctrl$x <- log_lambda
    ctrl$y <- beta_l2norm[, - inter]
    last_px <- log_lambda[length(log_lambda)]
    coef_idx <- which(beta_l2norm[nrow(beta_l2norm), - inter] > 0)
    last_py <- beta_l2norm[nrow(beta_l2norm), coef_idx + inter]
    ## main plot
    do.call(matplot, ctrl)
    text(x = rep(last_px, length(coef_idx)),
         y = last_py, label = coef_idx, cex = 0.5, pos = 2)
    if (x$specs$intercept) {
        lines(x = log_lambda, y = beta_l2norm[, 1L],
              col = "grey", lty = 2)
        text(x = last_px, y = beta_l2norm[nrow(beta_l2norm), 1L],
             label = "0", cex = 0.5, pos = 2, offset = 0.5)
    }
    ncov <- sapply(seq_along(log_lambda), function(k) {
        sum(beta_l2norm[k, - inter] > 0.0)
    })
    show_ncov <- c(1, which(diff(ncov) != 0) + 1L)
    ncov_x <- log_lambda[show_ncov]
    ## axis(1, at = ncov_x, tck = 0.01, labels = NA)
    axis(3, at = ncov_x, labels = ncov[show_ncov], tck = -0.01,
         cex.axis = 0.5, gap.axis = 0.01)
}
