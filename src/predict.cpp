//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2026 Eli Lilly and Company
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

#include <RcppEigen.h>
#include <abclass.h>

template <typename T_x>
Eigen::MatrixXd predict_prob(const T_x& x, abclass::ConstRefMatrixXd beta,
                             abclass::ConstRefMatrixXd offset,
                             const size_t loss_id,
                             const Rcpp::List& loss_params)
{
    const Eigen::Index k{beta.cols() + 1};
    const bool intercept{beta.rows() > x.cols()};
    switch (loss_id) {
    case 1: {
        abclass::LogitNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_prob(beta, x, offset);
    }
    case 2: {
        abclass::BoostNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        object.loss_fun_.set_inner_min(loss_params["boost_umin"]);
        return object.predict_prob(beta, x, offset);
    }
    case 3: {
        abclass::HBoostNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        object.loss_fun_.set_c(loss_params["lum_c"]);
        return object.predict_prob(beta, x, offset);
    }
    case 4: {
        abclass::LumNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        object.loss_fun_.set_ac(loss_params["lum_a"], loss_params["lum_c"]);
        return object.predict_prob(beta, x, offset);
    }
    case 5: {
        // "mlogit" loss: former Mlogit class, now an alias for LikeBoost
        abclass::LeBoostNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_prob(beta, x, offset);
    }
    case 6: {
        abclass::LeLogitNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_prob(beta, x, offset);
    }
    case 7: {
        abclass::LeBoostNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_prob(beta, x, offset);
    }
    default:
        break;
    }
    return Eigen::MatrixXd();
}

template <typename T_x>
Eigen::VectorXi predict_y(const T_x& x, abclass::ConstRefMatrixXd beta,
                           abclass::ConstRefMatrixXd offset,
                           const size_t loss_id)
{
    const Eigen::Index k{beta.cols() + 1};
    const bool intercept{beta.rows() > x.cols()};
    switch (loss_id) {
    case 1: {
        abclass::LogitNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_y(beta, x, offset);
    }
    case 2: {
        abclass::BoostNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_y(beta, x, offset);
    }
    case 3: {
        abclass::HBoostNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_y(beta, x, offset);
    }
    case 4: {
        abclass::LumNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_y(beta, x, offset);
    }
    case 5: {
        // "mlogit" loss: former Mlogit class, now an alias for LikeBoost
        abclass::LeBoostNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_y(beta, x, offset);
    }
    case 6: {
        abclass::LeLogitNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_y(beta, x, offset);
    }
    case 7: {
        abclass::LeBoostNet<T_x> object;
        object.data_.set_k(k);
        object.ctrl_.set_intercept(intercept);
        return object.predict_y(beta, x, offset);
    }
    default:
        break;
    }
    return Eigen::VectorXi();
}

template <typename T_x>
Eigen::MatrixXd predict_link(const T_x& x, abclass::ConstRefMatrixXd beta,
                              abclass::ConstRefMatrixXd offset)
{
    const Eigen::Index k{beta.cols() + 1};
    const bool intercept{beta.rows() > x.cols()};
    abclass::LogitNet<T_x> object;
    object.data_.set_k(k);
    object.ctrl_.set_intercept(intercept);
    return object.linear_score(beta, x, offset);
}

// [[Rcpp::export]]
Eigen::MatrixXd rcpp_pred_prob(const Eigen::MatrixXd& beta,
                               const Eigen::MatrixXd& x,
                               const Eigen::MatrixXd& offset,
                               const size_t loss_id,
                               const Rcpp::List& loss_params)
{
    return predict_prob(x, beta, offset, loss_id, loss_params);
}

// [[Rcpp::export]]
Eigen::MatrixXd rcpp_pred_prob_sp(const Eigen::MatrixXd& beta,
                                  const Eigen::SparseMatrix<double>& x,
                                  const Eigen::MatrixXd& offset,
                                  const size_t loss_id,
                                  const Rcpp::List& loss_params)
{
    return predict_prob(x, beta, offset, loss_id, loss_params);
}

// [[Rcpp::export]]
Eigen::VectorXi rcpp_pred_y(const Eigen::MatrixXd& beta,
                            const Eigen::MatrixXd& x,
                            const Eigen::MatrixXd& offset,
                            const size_t loss_id)
{
    return predict_y(x, beta, offset, loss_id);
}

// [[Rcpp::export]]
Eigen::VectorXi rcpp_pred_y_sp(const Eigen::MatrixXd& beta,
                               const Eigen::SparseMatrix<double>& x,
                               const Eigen::MatrixXd& offset,
                               const size_t loss_id)
{
    return predict_y(x, beta, offset, loss_id);
}

// [[Rcpp::export]]
Eigen::MatrixXd rcpp_pred_link(const Eigen::MatrixXd& beta,
                               const Eigen::MatrixXd& x,
                               const Eigen::MatrixXd& offset)
{
    return predict_link(x, beta, offset);
}

// [[Rcpp::export]]
Eigen::MatrixXd rcpp_pred_link_sp(const Eigen::MatrixXd& beta,
                                  const Eigen::SparseMatrix<double>& x,
                                  const Eigen::MatrixXd& offset)
{
    return predict_link(x, beta, offset);
}
