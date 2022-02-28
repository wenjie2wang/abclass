//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2022 Eli Lilly and Company
//
// This file is part of the R package abclass.
//
// The R package abclass is free software: You can redistribute it and/or
// modify it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or any later
// version (at your option). See the GNU General Public License at
// <https://www.gnu.org/licenses/> for details.
//
// The R package abclass is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//

#include <RcppArmadillo.h>
#include <abclass.h>
#include "export-helpers.h"

// [[Rcpp::export]]
Rcpp::List rcpp_hinge_boost_net(
    const arma::mat& x,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const arma::vec& weight,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int nfolds = 0,
    const bool stratified_cv = true,
    const unsigned int max_iter = 1e5,
    const double rel_tol = 1e-4,
    const bool varying_active_set = true,
    const double lum_c = 0.0,
    const unsigned int verbose = 0
    )
{
    abclass::HingeBoostNet object {
        x, y, lum_c, intercept, standardize, weight
    };
    return abclass_net_fit(object, y,
                           lambda, alpha, nlambda, lambda_min_ratio,
                           nfolds, stratified_cv, max_iter, rel_tol,
                           varying_active_set, verbose);
}
