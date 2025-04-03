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

// returns for et procedure
template <typename T_x>
inline Rcpp::List get_et_res(const abclass::Output<T_x>& object)
{
    return Rcpp::List::create(
        Rcpp::Named("nstages") = object.control_.et_nstages_,
        Rcpp::Named("selected") = abclass::arma2rvec(object.et_vs_),
        Rcpp::Named("l1_lambda0") =
            abclass::arma2rvec(object.et_l1_lambda0_vec_),
        Rcpp::Named("l1_lambda1") =
            abclass::arma2rvec(object.et_l1_lambda1_vec_));
}

// returns for cv procedure
template <typename T_x>
inline Rcpp::List get_cv_res(const abclass::Output<T_x>& object)
{
    return Rcpp::List::create(
        Rcpp::Named("nfolds") = object.control_.cv_nfolds_,
        Rcpp::Named("stratified") = object.control_.cv_stratified_,
        Rcpp::Named("alignment") = object.control_.cv_alignment_,
        Rcpp::Named("cv_accuracy") = object.cv_accuracy_,
        Rcpp::Named("cv_accuracy_mean") =
            abclass::arma2rvec(object.cv_accuracy_mean_),
        Rcpp::Named("cv_accuracy_sd") =
            abclass::arma2rvec(object.cv_accuracy_sd_));
}

// convert abclass::{Output,Control} to Rcpp::List
template <typename T_x>
inline Rcpp::List get_all_res(const abclass::Output<T_x>& object)
{
    Rcpp::List et_res, cv_res;
    if (object.control_.et_nstages_ > 0) {
        // et procedure
        et_res = get_et_res(object);
        // add estimates from cv
        if (object.control_.cv_nfolds_ > 0) {
            cv_res = get_cv_res(object);
        }
    } else {
        // add cv results
        if (object.control_.cv_nfolds_ > 0) {
            cv_res = get_cv_res(object);
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("optimization") = Rcpp::List::create(
            Rcpp::Named("loss") = abclass::arma2rvec(object.loss_),
            Rcpp::Named("penalty") = abclass::arma2rvec(object.penalty_),
            Rcpp::Named("objective") = abclass::arma2rvec(object.objective_),
            Rcpp::Named("n_iterations") = object.n_iter_
            ),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") =
            abclass::arma2rvec(object.control_.lambda_),
            Rcpp::Named("alpha") = object.control_.ridge_alpha_,
            Rcpp::Named("penalty_factor") =
            abclass::arma2rvec(object.control_.penalty_factor_),
            Rcpp::Named("lambda_max") = object.lambda_max_,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_,
            Rcpp::Named("ncv_kappa") = object.control_.ncv_kappa_,
            Rcpp::Named("ncv_gamma") = object.control_.ncv_gamma_,
            Rcpp::Named("gel_tau") = object.control_.gel_tau_,
            Rcpp::Named("mellowmax_omega") = object.control_.mellowmax_omega_
            ),
        Rcpp::Named("weights") =
        abclass::arma2rvec(object.data_->obs_weights_),
        Rcpp::Named("offset") = abclass::arma2rvec(object.data_->offset_),
        Rcpp::Named("cross_validation") = cv_res,
        Rcpp::Named("et") = et_res
        );
}

// convert the given control from Rcpp::List to abclass::Control
inline abclass::Control abclass_control(const Rcpp::List& control)
{
    abclass::Control ctrl {
        control["maxit"],
        control["epsilon"],
        control["verbose"]
    };
    ctrl.set_lower_limit(control["lower_limit"])->
        set_upper_limit(control["upper_limit"])->
        reg_path(control["nlambda"],
                 control["lambda_min_ratio"],
                 control["penalty_factor"],
                 control["varying_active_set"],
                 control["adjust_mm"])->
        reg_lambda(control["lambda"])->
        reg_ridge(control["alpha"],
                  control["lambda_max_alpha_min"])->
        reg_ncv(control["ncv_kappa"])->
        reg_gel(control["gel_tau"])->
        reg_mellowmax(control["mellowmax_omega"])->
        tune_cv(control["nfolds"],
                control["stratified"],
                control["alignment"])->
        tune_et(control["nstages"]);
    return ctrl;
}

// template interface
template <typename T>
Rcpp::List t_abclass_linear_fit(
    const T& x,
    const arma::uvec& y,
    const Rcpp::List& control
    )
{
    const size_t loss_id { control["loss_id"] };
    const size_t penalty_id { control["penalty_id"] };
    abclass::Control ctrl{abclass_control(control)};
    abclass::Data<T> train_data{x, y};
    train_data.set_obs_weights(control["weights"]);
    train_data.set_offset(control["offset"]);
    train_data.set_owl_reward(control["owl_reward"]);
    abclass::Output<T> res{
        abclass::abclass_linear_fit(train_data, ctrl, loss_id, penalty_id)};
    return get_all_res(res);
}

// [[Rcpp::export]]
Rcpp::List rcpp_abclass_linear_fit(
    const arma::mat& x,
    const arma::uvec& y,
    const Rcpp::List& control
    )
{
    return t_abclass_linear_fit<arma::mat>(x, y, control);
}

// [[Rcpp::export]]
Rcpp::List rcpp_abclass_linear_fit_sp(
    const arma::sp_mat& x,
    const arma::uvec& y,
    const Rcpp::List& control
    )
{
    return t_abclass_linear_fit<arma::sp_mat>(x, y, control);
}
