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

// template returns for Abclass objects
template <typename T>
inline Rcpp::List template_fit(T& object)
{
    if (object.control_.et_nstages_ > 0) {
        abclass::et_lambda(object);
        return Rcpp::List::create(
            Rcpp::Named("coefficients") = object.coef_.slice(0),
            Rcpp::Named("weight") =
            abclass::arma2rvec(object.control_.obs_weight_),
            Rcpp::Named("et") = Rcpp::List::create(
                Rcpp::Named("nstages") = object.control_.et_nstages_,
                Rcpp::Named("selected") = abclass::arma2rvec(object.et_vs_)
                ),
            Rcpp::Named("regularization") = Rcpp::List::create(
                Rcpp::Named("alpha") = object.control_.alpha_,
                Rcpp::Named("group_weight") =
                abclass::arma2rvec(object.control_.group_weight_),
                Rcpp::Named("dgamma") = object.control_.dgamma_,
                Rcpp::Named("gamma") = object.control_.gamma_
                )
            );
    }
    Rcpp::List cv_res;
    if (object.control_.cv_nfolds_ > 0) {
        arma::uvec strata;
        if (object.control_.cv_stratified_) {
            strata = object.y_;
        }
        abclass::cv_lambda(object, strata);
        cv_res = Rcpp::List::create(
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
    object.fit();
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("weight") = abclass::arma2rvec(object.control_.obs_weight_),
        Rcpp::Named("cross_validation") = cv_res,
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") =
            abclass::arma2rvec(object.control_.lambda_),
            Rcpp::Named("lambda_max") = object.lambda_max_,
            Rcpp::Named("alpha") = object.control_.alpha_,
            Rcpp::Named("group_weight") =
            abclass::arma2rvec(object.control_.group_weight_),
            Rcpp::Named("dgamma") = object.control_.dgamma_,
            Rcpp::Named("gamma") = object.control_.gamma_
            ),
        Rcpp::Named("loss_wo_penalty") = abclass::arma2rvec(
            object.loss_wo_penalty_),
        Rcpp::Named("penalty") = abclass::arma2rvec(object.penalty_)
        );
}

// convert the given control from Rcpp::List to abclass::Control
inline abclass::Control conv_control(const Rcpp::List& control)
{
    abclass::Control ctrl {
        control["maxit"], control["epsilon"],
        control["standardize"], control["verbose"]
    };
    ctrl.set_intercept(control["intercept"])->
        set_weight(control["weight"])->
        reg_path(control["nlambda"],
                 control["lambda_min_ratio"],
                 control["varying_active_set"])->
        reg_path(control["lambda"])->
        reg_net(control["alpha"])->
        reg_group(control["group_weight"],
                  control["dgamma"])->
        tune_cv(control["nfolds"],
                control["stratified"],
                control["alignment"])->
        tune_et(control["nstages"]);
    return ctrl;
}
