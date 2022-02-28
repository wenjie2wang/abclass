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

// [[Rcpp::export]]
arma::mat rcpp_logistic_predict_prob(const arma::mat& beta,
                                     const arma::mat& x)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::LogisticNet object { k };
    return object.predict_prob(beta, x);
}

// [[Rcpp::export]]
arma::uvec rcpp_logistic_predict_y(const arma::mat& beta,
                                   const arma::mat& x)
{
    return arma::index_max(rcpp_logistic_predict_prob(beta, x), 1);
}

// [[Rcpp::export]]
arma::mat rcpp_boost_predict_prob(const arma::mat& beta,
                                  const arma::mat& x,
                                  const double inner_min)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::BoostNet object { k };
    object.set_inner_min(inner_min);
    return object.predict_prob(beta, x);
}

// [[Rcpp::export]]
arma::uvec rcpp_boost_predict_y(const arma::mat& beta,
                                const arma::mat& x,
                                const double inner_min)
{
    return arma::index_max(rcpp_boost_predict_prob(beta, x, inner_min), 1);
}

// [[Rcpp::export]]
arma::mat rcpp_hinge_boost_predict_prob(const arma::mat& beta,
                                        const arma::mat& x,
                                        const double lum_c)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::HingeBoostNet object { k };
    object.set_lum_c(lum_c);
    return object.predict_prob(beta, x);
}

// [[Rcpp::export]]
arma::uvec rcpp_hinge_boost_predict_y(const arma::mat& beta,
                                      const arma::mat& x,
                                      const double lum_c)
{
    return arma::index_max(rcpp_hinge_boost_predict_prob(beta, x, lum_c), 1);
}

// [[Rcpp::export]]
arma::mat rcpp_lum_predict_prob(const arma::mat& beta,
                                const arma::mat& x,
                                const double lum_a,
                                const double lum_c)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::LumNet object { k };
    object.set_lum_parameters(lum_a, lum_c);
    return object.predict_prob(beta, x);
}

// [[Rcpp::export]]
arma::uvec rcpp_lum_predict_y(const arma::mat& beta,
                              const arma::mat& x,
                              const double lum_a,
                              const double lum_c)
{
    return arma::index_max(rcpp_lum_predict_prob(beta, x, lum_a, lum_c), 1);
}
