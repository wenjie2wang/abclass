//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2024 Eli Lilly and Company
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

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <abclass.h>
#include "template_helpers.h"

// construct control for ranking
inline abclass::Control abrank_control(const Rcpp::List& control)
{
    abclass::Control ctrl {
        control["maxit"], control["epsilon"],
        control["standardize"], control["verbose"]
    };
    ctrl.set_intercept(false)->
        set_weight(control["weight"])->
        rank(control["query_weight"],
             control["delta_weight"],
             control["delta_adaptive"],
             control["delta_maxit"])->
        set_offset<arma::vec>(control["offset"])->
        reg_path(control["nlambda"],
                 control["lambda_min_ratio"],
                 control["varying_active_set"])->
        reg_path(control["lambda"])->
        reg_net(control["alpha"])->
        tune_cv(control["cv_metric"])->
        tune_et(control["nstages"]);
    return ctrl;
}

template <typename T>
inline Rcpp::List template_abrank_fit(T& object)
{
    if (object.abc_.control_.et_nstages_ > 0) {
        abclass::et_lambda(object.abc_);
        return Rcpp::List::create(
            Rcpp::Named("coefficients") = object.abc_.coef_.slice(0),
            Rcpp::Named("weight") =
            abclass::arma2rvec(object.abc_.control_.obs_weight_),
            Rcpp::Named("et") = Rcpp::List::create(
                Rcpp::Named("nstages") = object.abc_.control_.et_nstages_,
                Rcpp::Named("selected") = abclass::arma2rvec(object.abc_.et_vs_)
                ),
            Rcpp::Named("regularization") = Rcpp::List::create(
                Rcpp::Named("alpha") = object.abc_.control_.alpha_
                )
            );
    }
    Rcpp::List cv_res;
    if (object.abc_.control_.cv_nfolds_ == 1) {
        cv_res = Rcpp::List::create(
            Rcpp::Named("cv_metric") = 1,
            Rcpp::Named("cv_recall") = object.cv_recall()
            );
    }
    if (object.abc_.control_.cv_nfolds_ == 2) {
        cv_res = Rcpp::List::create(
            Rcpp::Named("cv_metric") = 2,
            Rcpp::Named("cv_delta_recall") = object.cv_delta_recall()
            );
    }
    if (object.abc_.control_.cv_nfolds_ >= 3) {
        abclass::cv_lambda(object.abc_);
        cv_res = Rcpp::List::create(
            Rcpp::Named("nfolds") = object.abc_.control_.cv_nfolds_,
            Rcpp::Named("alignment") = object.abc_.control_.cv_alignment_,
            Rcpp::Named("cv_accuracy") = object.abc_.cv_accuracy_,
            Rcpp::Named("cv_accuracy_mean") =
            abclass::arma2rvec(object.abc_.cv_accuracy_mean_),
            Rcpp::Named("cv_accuracy_sd") =
            abclass::arma2rvec(object.abc_.cv_accuracy_sd_)
            );
    }
    object.fit();
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.abc_.coef_,
        Rcpp::Named("weight") =
        abclass::arma2rvec(object.abc_.control_.obs_weight_),
        Rcpp::Named("cross_validation") = cv_res,
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") =
            abclass::arma2rvec(object.abc_.control_.lambda_),
            Rcpp::Named("lambda_max") = object.abc_.lambda_max_,
            Rcpp::Named("alpha") = object.abc_.control_.alpha_
            ),
        Rcpp::Named("loss_wo_penalty") = abclass::arma2rvec(
            object.abc_.loss_wo_penalty_),
        Rcpp::Named("penalty") = abclass::arma2rvec(object.abc_.penalty_)
        );
}

// [[Rcpp::export]]
Rcpp::List rcpp_abrank_fit(
    const arma::mat& x,
    const arma::vec& y,
    const arma::uvec& qid,
    const Rcpp::List& control
    )
{
    abclass::Control ctrl { abrank_control(control) };
    const size_t loss_id { control["loss_id"] };
    switch (loss_id) {
        case 1: {
            abclass::LogisticRank<arma::mat> object { x, y, qid, ctrl };
            return template_abrank_fit(object);
        }
        case 2: {
            abclass::BoostRank<arma::mat> object { x, y, qid, ctrl };
            object.abc_.loss_.set_inner_min(control["boost_umin"]);
            return template_abrank_fit(object);
        }
        case 3: {
            abclass::HingeBoostRank<arma::mat> object { x, y, qid, ctrl };
            object.abc_.loss_.set_c(control["lum_c"]);
            return template_abrank_fit(object);
        }
        case 4: {
            abclass::LumRank<arma::mat> object { x, y, qid, ctrl };
            object.abc_.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_abrank_fit(object);
        }
        default:
            break;
    }
    return Rcpp::List();
}

// [[Rcpp::export]]
Rcpp::List rcpp_query_delta_weight(const arma::vec& y,
                                   const arma::vec& pred)
{
    abclass::Query<arma::mat> q_obj { y, true };
    arma::uvec rev_idx { q_obj.get_rev_idx() };
    arma::uvec i_vec { q_obj.pair_i_.elem(rev_idx) + 1 };
    arma::uvec j_vec { q_obj.pair_j_.elem(rev_idx) + 1 };
    q_obj.compute_max_dcg();
    arma::vec delta_weight;
    if (pred.is_empty()) {
        delta_weight = q_obj.delta_ndcg();
    } else {
        delta_weight = q_obj.delta_ndcg(pred, false);
    }
    delta_weight = delta_weight.elem(rev_idx);
    return Rcpp::List::create(
        Rcpp::Named("lambda_weight") = abclass::arma2rvec(delta_weight),
        Rcpp::Named("i") = abclass::arma2rvec(i_vec),
        Rcpp::Named("j") = abclass::arma2rvec(j_vec)
        );
}
