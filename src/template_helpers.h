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

// template returns for et procedure
template <typename T>
inline Rcpp::List get_et_res(const T& object)
{
    return Rcpp::List::create(
        Rcpp::Named("nstages") = object.control_.et_nstages_,
        Rcpp::Named("selected") = abclass::arma2rvec(object.et_vs_),
        Rcpp::Named("l1_lambda0") =
        abclass::arma2rvec(object.et_l1_lambda0_vec_),
        Rcpp::Named("l1_lambda1") =
        abclass::arma2rvec(object.et_l1_lambda1_vec_)
        );
}

// template returns for cv procedure
template <typename T>
inline Rcpp::List get_cv_res(const T& object)
{
    return Rcpp::List::create(
        Rcpp::Named("nfolds") = object.control_.cv_nfolds_,
        Rcpp::Named("stratified") = object.control_.cv_stratified_,
        Rcpp::Named("alignment") = object.control_.cv_alignment_,
        Rcpp::Named("cv_accuracy") = object.cv_accuracy_,
        Rcpp::Named("cv_accuracy_mean") =
        abclass::arma2rvec(object.cv_accuracy_mean_),
        Rcpp::Named("cv_accuracy_sd") =
        abclass::arma2rvec(object.cv_accuracy_sd_)
        );
}

// template returns for Abclass objects
template <typename T>
inline Rcpp::List template_fit(T& object)
{
    Rcpp::List et_res, cv_res;
    if (object.control_.et_nstages_ > 0) {
        // et procedure
        abclass::et_lambda(object);
        et_res = get_et_res(object);
        // add estimates from cv
        if (object.control_.cv_nfolds_ > 0) {
            abclass::et_cv_accuracy(object);
            cv_res = get_cv_res(object);
        }
    } else {
        // main fit
        object.fit();
        // add cv results
        if (object.control_.cv_nfolds_ > 0) {
            abclass::cv_lambda(object);
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
        abclass::arma2rvec(object.control_.obs_weight_),
        Rcpp::Named("offset") = abclass::arma2rvec(object.control_.offset_),
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
        control["standardize"],
        control["verbose"]
    };
    ctrl.set_intercept(control["intercept"])->
        set_weight(control["weights"])->
        set_offset(control["offset"])->
        set_lower_limit(control["lower_limit"])->
        set_upper_limit(control["upper_limit"])->
        set_owl_reward(control["owl_reward"])->
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
