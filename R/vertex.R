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

##' Simplex Vertices for The Angle-Based Classification
##'
##' @param k Number of classes, a positive integer that is greater than one.
##'
##' @return A \code{(k-1)} by \code{k} matrix that consists of vertices in
##'     columns.
##'
##' @references
##'
##' Lange, K., & Tong Wu, Tong (2008). An MM algorithm for multicategory vertex
##' discriminant analysis. Journal of Computational and Graphical Statistics,
##' 17(3), 527--544.
##'
##' @export
vertex <- function(k)
{
    rcpp_vertex(as.integer(k))
}
