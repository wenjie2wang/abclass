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

## convert all the categories to {0, ..., k - 1}
## or {1, ..., k} if zero_based is FALSE
cat2z <- function(y, zero_based = TRUE) {
    fac_y <- as.factor(y)
    list(y = as.integer(fac_y) - as.integer(zero_based),
         label = levels(fac_y),
         class_y = class(y))
}

## reverse convert
z2cat <- function(y, cat_y, zero_based = TRUE) {
    out <- cat_y$label[y + as.integer(zero_based)]
    switch(cat_y$class_y,
           "integer" = as.integer(out),
           "numeric" = as.numeric(out),
           "factor" = factor(out, levels = cat_y$label),
           "character" = as.character(out))
}

## convert null to numeric(0)
null2num0 <- function(x) {
    if (is.null(x)) {
        return(numeric(0))
    }
    x
}

## convert null to numeric(0)
null2mat0 <- function(x) {
    if (is.null(x)) {
        return(matrix(numeric(0)))
    }
    as.matrix(x)
}

## return 0 if null, x otherwise
null0 <- function(x) {
    if (is.null(x)) {
        return(0)
    }
    x
}

## select lambda's from the solution path
select_lambda <- function(cv_mean, cv_sd) {
    ## the cv_mean and cv_sd correspond to the decreasing lambda sequence
    cv_min_idx <- which.max(cv_mean)
    cv_min <- cv_mean[cv_min_idx]
    cv_min_sd <- cv_sd[cv_min_idx]
    cv_1se_idx <- min(which(cv_mean >= cv_min - cv_min_sd))
    list(cv_min = cv_min_idx,
         cv_1se = cv_1se_idx)
}

## function formal names (formalArgs() from the methods package)
formal_names <- function(def) {
    names(formals(def, envir = parent.frame()))
}

## simplified version of utils::modifyList with keep.null = TRUE
modify_list <- function (x, val) {
    stopifnot(is.list(x), is.list(val))
    xnames <- names(x)
    vnames <- names(val)
    vnames <- vnames[nzchar(vnames)]
    for (v in vnames) {
        x[v] <- if (v %in% xnames && is.list(x[[v]]) && is.list(val[[v]]))
                    list(modify_list(x[[v]], val[[v]]))
                else val[v]
    }
    x
}

## L2-norm square
l2norm2 <- function(x)
{
    sum(x ^ 2)
}
## L2-norm
l2norm <- function(x)
{
    sqrt(l2norm2(x))
}
## row-wise L2-norms
rowL2norms <- function(x)
{
    apply(x, 1L, l2norm)
}
## sum of row-wise L2-norms
rowL2sums <- function(x)
{
    sum(rowL2norms(x))
}
## row-wise sup-norm
rowSupnorms <- function(x) {
    apply(abs(x), 1L, max)
}

## check if the suggested package is available
suggest_pkg <- function(pkg_name)
{
    if (! requireNamespace(pkg_name, quietly = TRUE)) {
        stop(sprintf("The package '%s' is needed but not available.",
                     pkg_name), call. = FALSE)
    }
    invisible()
}

## get the default values of arguments
default_args <- function(fun)
{
    flist <- formals(fun)
    lapply(flist, function(a) {
        if (is.symbol(a)) {
            return(NULL)
        }
        if (is.language(a)) {
            return(eval(a))
        }
        a
    })
}
