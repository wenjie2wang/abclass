//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2025 Eli Lilly and Company
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
                       const arma::mat& offset,
                       const size_t loss_id,
                       const Rcpp::List& loss_params)
{
    const unsigned int k { beta.n_cols + 1 };
    switch (loss_id) {
        case 1:
        {
            abclass::AbclassLinear<abclass::Logistic, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x, offset);
        }
        case 2:
        {
            abclass::AbclassLinear<abclass::Boost, T_x> object { k };
            object.loss_fun_.set_inner_min(loss_params["boost_umin"]);
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x, offset);
        }
        case 3:
        {
            abclass::AbclassLinear<abclass::HingeBoost, T_x> object { k };
            object.loss_fun_.set_c(loss_params["lum_c"]);
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x, offset);
        }
        case 4:
        {
            abclass::AbclassLinear<abclass::Lum, T_x> object { k };
            object.loss_fun_.set_ac(loss_params["lum_a"], loss_params["lum_c"]);
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x, offset);
        }
        case 5:
        {
            abclass::AbclassLinear<abclass::Mlogit, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x, offset);
        }
        case 6:
        {
            abclass::AbclassLinear<abclass::LikeLogistic, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x, offset);
        }
        case 7:
        {
            abclass::AbclassLinear<abclass::LikeBoost, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x, offset);
        }
        case 8:
        {
            abclass::AbclassLinear<abclass::LikeHingeBoost, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x, offset);
        }
        case 9:
        {
            abclass::AbclassLinear<abclass::LikeLum, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_prob(beta, x, offset);
        }
        default:
            break;
    }
    return arma::mat();
}

template <typename T_x>
arma::uvec predict_y(const T_x& x,
                     const arma::mat& beta,
                     const arma::mat& offset,
                     const size_t loss_id)
{
    const unsigned int k { beta.n_cols + 1 };
    switch (loss_id) {
        case 1:
        {
            abclass::AbclassLinear<abclass::Logistic, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x, offset);
        }
        case 2:
        {
            abclass::AbclassLinear<abclass::Boost, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x, offset);
        }
        case 3:
        {
            abclass::AbclassLinear<abclass::HingeBoost, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x, offset);
        }
        case 4:
        {
            abclass::AbclassLinear<abclass::Lum, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x, offset);
        }
        case 5:
        {
            abclass::AbclassLinear<abclass::Mlogit, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x, offset);
        }
        case 6:
        {
            abclass::AbclassLinear<abclass::LikeLogistic, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x, offset);
        }
        case 7:
        {
            abclass::AbclassLinear<abclass::LikeBoost, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x, offset);
        }
        case 8:
        {
            abclass::AbclassLinear<abclass::LikeHingeBoost, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x, offset);
        }
        case 9:
        {
            abclass::AbclassLinear<abclass::LikeLum, T_x> object { k };
            object.set_intercept(beta.n_rows > x.n_cols);
            return object.predict_y(beta, x, offset);
        }
        default:
            break;
    }
    return arma::uvec();
}

template <typename T_x>
arma::mat predict_link(const T_x& x,
                       const arma::mat& beta,
                       const arma::mat& offset)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::AbclassLinear<abclass::Logistic, T_x> object { k };
    object.set_intercept(beta.n_rows > x.n_cols);
    return object.linear_score(beta, x, offset);
}


// [[Rcpp::export]]
arma::mat rcpp_pred_prob(const arma::mat& beta,
                         const arma::mat& x,
                         const arma::mat& offset,
                         const size_t loss_id,
                         const Rcpp::List& loss_params)
{
    return predict_prob(x, beta, offset, loss_id, loss_params);
}

// [[Rcpp::export]]
arma::mat rcpp_pred_prob_sp(const arma::mat& beta,
                            const arma::sp_mat& x,
                            const arma::mat& offset,
                            const size_t loss_id,
                            const Rcpp::List& loss_params)
{
    return predict_prob(x, beta, offset, loss_id, loss_params);
}

// [[Rcpp::export]]
arma::uvec rcpp_pred_y(const arma::mat& beta,
                       const arma::mat& x,
                       const arma::mat& offset,
                       const size_t loss_id)
{
    return predict_y(x, beta, offset, loss_id);
}

// [[Rcpp::export]]
arma::uvec rcpp_pred_y_sp(const arma::mat& beta,
                          const arma::sp_mat& x,
                          const arma::mat& offset,
                          const size_t loss_id)
{
    return predict_y(x, beta, offset, loss_id);
}

// [[Rcpp::export]]
arma::mat rcpp_pred_link(const arma::mat& beta,
                         const arma::mat& x,
                         const arma::mat& offset)
{
    return predict_link(x, beta, offset);
}

// [[Rcpp::export]]
arma::mat rcpp_pred_link_sp(const arma::mat& beta,
                            const arma::sp_mat& x,
                            const arma::mat& offset)
{
    return predict_link(x, beta, offset);
}
