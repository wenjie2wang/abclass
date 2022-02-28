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

## convert all the categories to {0, ..., k - 1}
cat2z <- function(y) {
    fac_y <- as.factor(y)
    list(y = as.integer(fac_y) - 1L,
         label = levels(fac_y),
         class_y = class(y))
}

## reverse convert
z2cat <- function(y, cat_y) {
    out <- cat_y$label[y + 1L]
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
