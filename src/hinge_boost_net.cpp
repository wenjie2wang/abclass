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

template <typename T>
Rcpp::List hinge_boost_net(
    const T& x,
    const arma::uvec& y,
    const abclass::Control& control,
    const double lum_c
    )
{
    abclass::HingeBoostNet<T> object { x, y, control };
    object.loss_.set_c(lum_c);
    return abclass_net_fit(object);
}

// [[Rcpp::export]]
Rcpp::List r_hinge_boost_net(
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
    const unsigned int alignment = 0,
    const unsigned int maxit = 1e5,
    const double epsilon = 1e-3,
    const bool varying_active_set = true,
    const double lum_c = 0.0,
    const unsigned int verbose = 0
    )
{
    abclass::Control control { maxit, epsilon, standardize, verbose };
    control.set_intercept(intercept)->
        set_weight(weight)->
        reg_path(nlambda, lambda_min_ratio, varying_active_set)->
        reg_path(lambda)->
        reg_net(alpha)->
        tune_cv(nfolds, stratified_cv, alignment);
    return hinge_boost_net<arma::mat>(x, y, control, lum_c);
}

// [[Rcpp::export]]
Rcpp::List r_hinge_boost_net_sp(
    const arma::sp_mat& x,
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
    const unsigned int alignment = 0,
    const unsigned int maxit = 1e5,
    const double epsilon = 1e-3,
    const bool varying_active_set = true,
    const double lum_c = 0.0,
    const unsigned int verbose = 0
    )
{
    abclass::Control control { maxit, epsilon, standardize, verbose };
    control.set_intercept(intercept)->
        set_weight(weight)->
        reg_path(nlambda, lambda_min_ratio, varying_active_set)->
        reg_path(lambda)->
        reg_net(alpha)->
        tune_cv(nfolds, stratified_cv, alignment);
    return hinge_boost_net<arma::sp_mat>(x, y, control, lum_c);
}
