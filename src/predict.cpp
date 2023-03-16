//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2023 Eli Lilly and Company
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

template <typename T_x>
arma::mat predict_prob(const T_x& x,
                       const arma::mat& beta,
                       const size_t loss_id,
                       const Rcpp::List& loss_params)
{
    const unsigned int k { beta.n_cols + 1 };
    switch (loss_id) {
        case 1:
        {
            abclass::Abclass<abclass::Logistic, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x);
        }
        case 2:
        {
            abclass::Abclass<abclass::Boost, T_x> object { k };
            object.loss_.set_inner_min(loss_params["boost_umin"]);
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x);
        }
        case 3:
        {
            abclass::Abclass<abclass::HingeBoost, T_x> object { k };
            object.loss_.set_c(loss_params["lum_c"]);
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x);
        }
        case 4:
        {
            abclass::Abclass<abclass::Lum, T_x> object { k };
            object.loss_.set_ac(loss_params["lum_a"], loss_params["lum_c"]);
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x);
        }
        default:
            break;
    }
    return arma::mat();
}

template <typename T_x>
arma::uvec predict_y(const T_x& x,
                     const arma::mat& beta,
                     const size_t loss_id)
{
    const unsigned int k { beta.n_cols + 1 };
    switch (loss_id) {
        case 1:
        {
            abclass::Abclass<abclass::Logistic, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x);
        }
        case 2:
        {
            abclass::Abclass<abclass::Boost, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x);
        }
        case 3:
        {
            abclass::Abclass<abclass::HingeBoost, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x);
        }
        case 4:
        {
            abclass::Abclass<abclass::Lum, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x);
        }
        default:
            break;
    }
    return arma::uvec();
}

// [[Rcpp::export]]
arma::mat rcpp_pred_prob(const arma::mat& beta,
                         const arma::mat& x,
                         const size_t loss_id,
                         const Rcpp::List& loss_params)
{
    return predict_prob(x, beta, loss_id, loss_params);
}

// [[Rcpp::export]]
arma::mat rcpp_pred_prob_sp(const arma::mat& beta,
                            const arma::sp_mat& x,
                            const size_t loss_id,
                            const Rcpp::List& loss_params)
{
    return predict_prob(x, beta, loss_id, loss_params);
}

// [[Rcpp::export]]
arma::uvec rcpp_pred_y(const arma::mat& beta,
                       const arma::mat& x,
                       const size_t loss_id)
{
    return predict_y(x, beta, loss_id);
}

// [[Rcpp::export]]
arma::uvec rcpp_pred_y_sp(const arma::mat& beta,
                          const arma::sp_mat& x,
                          const size_t loss_id)
{
    return predict_y(x, beta, loss_id);
}
