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

template <typename T_loss, typename T_x>
arma::mat predict_prob(const arma::mat& beta, const T_x& x)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::Abclass<T_loss, T_x> object { k };
    object.set_intercept(beta.n_rows > x.n_cols);
    return object.predict_prob(beta, x);
}

template <typename T_loss, typename T_x>
arma::uvec predict_y(const arma::mat& beta, const T_x& x)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::Abclass<T_loss, T_x> object { k };
    object.set_intercept(beta.n_rows > x.n_cols);
    return object.predict_y(beta, x);
}

// logistic ==================================================================
// [[Rcpp::export]]
arma::mat r_logistic_pred_prob(const arma::mat& beta,
                                  const arma::mat& x)
{
    return predict_prob<abclass::Logistic, arma::mat>(beta, x);
}
// [[Rcpp::export]]
arma::mat r_logistic_pred_prob_sp(const arma::mat& beta,
                                  const arma::sp_mat& x)
{
    return predict_prob<abclass::Logistic, arma::sp_mat>(beta, x);
}
// [[Rcpp::export]]
arma::uvec r_logistic_pred_y(const arma::mat& beta,
                             const arma::mat& x)
{
    return predict_y<abclass::Logistic, arma::mat>(beta, x);
}
// [[Rcpp::export]]
arma::uvec r_logistic_pred_y_sp(const arma::mat& beta,
                                const arma::sp_mat& x)
{
    return predict_y<abclass::Logistic, arma::sp_mat>(beta, x);
}

// boost =====================================================================
// [[Rcpp::export]]
arma::mat r_boost_pred_prob(const arma::mat& beta,
                            const arma::mat& x)
{
    return predict_prob<abclass::Boost, arma::mat>(beta, x);
}
// [[Rcpp::export]]
arma::mat r_boost_pred_prob_sp(const arma::mat& beta,
                               const arma::sp_mat& x)
{
    return predict_prob<abclass::Boost, arma::sp_mat>(beta, x);
}
// [[Rcpp::export]]
arma::uvec r_boost_pred_y(const arma::mat& beta,
                          const arma::mat& x)
{
    return predict_y<abclass::Boost, arma::mat>(beta, x);
}
// [[Rcpp::export]]
arma::uvec r_boost_pred_y_sp(const arma::mat& beta,
                             const arma::sp_mat& x)
{
    return predict_y<abclass::Boost, arma::sp_mat>(beta, x);
}

// hinge-boost ===============================================================
// [[Rcpp::export]]
arma::mat r_hinge_boost_pred_prob(const arma::mat& beta,
                                  const arma::mat& x)
{
    return predict_prob<abclass::HingeBoost, arma::mat>(beta, x);
}
// [[Rcpp::export]]
arma::mat r_hinge_boost_pred_prob_sp(const arma::mat& beta,
                                     const arma::sp_mat& x)
{
    return predict_prob<abclass::HingeBoost, arma::sp_mat>(beta, x);
}
// [[Rcpp::export]]
arma::uvec r_hinge_boost_pred_y(const arma::mat& beta,
                                const arma::mat& x)
{
    return predict_y<abclass::HingeBoost, arma::mat>(beta, x);
}
// [[Rcpp::export]]
arma::uvec r_hinge_boost_pred_y_sp(const arma::mat& beta,
                                   const arma::sp_mat& x)
{
    return predict_y<abclass::HingeBoost, arma::sp_mat>(beta, x);
}

// lum =======================================================================
// [[Rcpp::export]]
arma::mat r_lum_pred_prob(const arma::mat& beta,
                          const arma::mat& x)
{
    return predict_prob<abclass::Lum, arma::mat>(beta, x);
}
// [[Rcpp::export]]
arma::mat r_lum_pred_prob_sp(const arma::mat& beta,
                             const arma::sp_mat& x)
{
    return predict_prob<abclass::Lum, arma::sp_mat>(beta, x);
}
// [[Rcpp::export]]
arma::uvec r_lum_pred_y(const arma::mat& beta,
                        const arma::mat& x)
{
    return predict_y<abclass::Lum, arma::mat>(beta, x);
}
// [[Rcpp::export]]
arma::uvec r_lum_pred_y_sp(const arma::mat& beta,
                           const arma::sp_mat& x)
{
    return predict_y<abclass::Lum, arma::sp_mat>(beta, x);
}
