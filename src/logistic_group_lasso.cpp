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
#include <abclass/LogisticGroupLasso.h>
#include "export-helpers.h"

template <typename T>
Rcpp::List logistic_glasso(
    const T& x,
    const arma::uvec& y,
    const abclass::Control& control,
    const bool main_fit
    )
{
    abclass::LogisticGroupLasso<T> object { x, y, control };
    return template_fit(object, main_fit);
}

// [[Rcpp::export]]
Rcpp::List r_logistic_glasso(
    const arma::mat& x,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const arma::vec& group_weight,
    const arma::vec& weight,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int maxit = 1e5,
    const double epsilon = 1e-3,
    const bool varying_active_set = true,
    const unsigned int verbose = 0,
    const unsigned int nfolds = 0,
    const bool stratified = true,
    const unsigned int alignment = 0,
    const unsigned int nstages = 0,
    const bool main_fit = true
    )
{
    abclass::Control control { maxit, epsilon, standardize, verbose };
    control.set_intercept(intercept)->
        set_weight(weight)->
        reg_path(nlambda, lambda_min_ratio, varying_active_set)->
        reg_path(lambda)->
        reg_net(alpha)->
        reg_group(group_weight)->
        tune_cv(nfolds, stratified, alignment)->
        tune_et(nstages);
    return logistic_glasso<arma::mat>(x, y, control, main_fit);
}

// [[Rcpp::export]]
Rcpp::List r_logistic_glasso_sp(
    const arma::sp_mat& x,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const arma::vec& group_weight,
    const arma::vec& weight,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int maxit = 1e5,
    const double epsilon = 1e-3,
    const bool varying_active_set = true,
    const unsigned int verbose = 0,
    const unsigned int nfolds = 0,
    const bool stratified = true,
    const unsigned int alignment = 0,
    const unsigned int nstages = 0,
    const bool main_fit = true
    )
{
    abclass::Control control { maxit, epsilon, standardize, verbose };
    control.set_intercept(intercept)->
        set_weight(weight)->
        reg_path(nlambda, lambda_min_ratio, varying_active_set)->
        reg_path(lambda)->
        reg_net(alpha)->
        reg_group(group_weight)->
        tune_cv(nfolds, stratified, alignment)->
        tune_et(nstages);
    return logistic_glasso<arma::sp_mat>(x, y, control, main_fit);
}
