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

// convert std::vector<Eigen::MatrixXd> to a 3D R array (rows x cols x slices)
inline SEXP coef_to_array(const std::vector<Eigen::MatrixXd>& coef)
{
    if (coef.empty()) {
        return R_NilValue;
    }
    const int nrow = static_cast<int>(coef[0].rows());
    const int ncol = static_cast<int>(coef[0].cols());
    const int nslice = static_cast<int>(coef.size());
    Rcpp::NumericVector arr(Rcpp::Dimension(nrow, ncol, nslice));
    for (int s = 0; s < nslice; ++s) {
        for (int j = 0; j < ncol; ++j) {
            for (int i = 0; i < nrow; ++i) {
                arr[s * nrow * ncol + j * nrow + i] = coef[s](i, j);
            }
        }
    }
    return arr;
}

// returns for et procedure
template <typename T_abclass>
inline Rcpp::List get_et_res(const T_abclass& object)
{
    return Rcpp::List::create(
        Rcpp::Named("nstages") = object.ctrl_.et_nstages,
        Rcpp::Named("selected") = Rcpp::wrap(object.result_.et_vs_),
        Rcpp::Named("l1_lambda0") =
            abclass::eigen2rvec(object.result_.et_l1_lambda0_vec_),
        Rcpp::Named("l1_lambda1") =
            abclass::eigen2rvec(object.result_.et_l1_lambda1_vec_));
}

// returns for cv procedure
template <typename T_abclass>
inline Rcpp::List get_cv_res(const T_abclass& object)
{
    return Rcpp::List::create(
        Rcpp::Named("nfolds") = object.ctrl_.cv_nfolds,
        Rcpp::Named("stratified") = object.ctrl_.cv_stratified,
        Rcpp::Named("alignment") = object.ctrl_.cv_alignment,
        Rcpp::Named("cv_accuracy") = Rcpp::wrap(object.tuning_.cv_accuracy_),
        Rcpp::Named("cv_accuracy_mean") =
            abclass::eigen2rvec(object.tuning_.cv_accuracy_mean_),
        Rcpp::Named("cv_accuracy_sd") =
            abclass::eigen2rvec(object.tuning_.cv_accuracy_sd_));
}

// convert a fitted abclass object to Rcpp::List
template <typename T_abclass>
inline Rcpp::List get_all_res(const T_abclass& object)
{
    Rcpp::List et_res, cv_res;
    if (object.ctrl_.et_nstages > 0) {
        // et procedure
        et_res = get_et_res(object);
        // add estimates from cv
        if (object.ctrl_.cv_nfolds > 0) {
            cv_res = get_cv_res(object);
        }
    } else {
        // add cv results
        if (object.ctrl_.cv_nfolds > 0) {
            cv_res = get_cv_res(object);
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = coef_to_array(object.result_.coef_),
        Rcpp::Named("optimization") = Rcpp::List::create(
            Rcpp::Named("loss") = abclass::eigen2rvec(object.result_.loss_),
            Rcpp::Named("penalty") =
            abclass::eigen2rvec(object.result_.penalty_),
            Rcpp::Named("objective") =
            abclass::eigen2rvec(object.result_.objective_),
            Rcpp::Named("n_iterations") = object.result_.n_iter_
            ),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") =
            abclass::eigen2rvec(object.ctrl_.lambda),
            Rcpp::Named("alpha") = object.ctrl_.ridge_alpha,
            Rcpp::Named("penalty_factor") =
            abclass::eigen2rvec(object.ctrl_.penalty_factor),
            Rcpp::Named("lambda_max") = object.result_.lambda_max_,
            Rcpp::Named("l1_lambda_max") = object.result_.l1_lambda_max_,
            Rcpp::Named("ncv_kappa") = object.ctrl_.ncv_kappa,
            Rcpp::Named("ncv_gamma") = object.ctrl_.ncv_gamma,
            Rcpp::Named("gel_tau") = object.ctrl_.gel_tau
            ),
        Rcpp::Named("weights") =
        abclass::eigen2rvec(object.data_.weights_),
        Rcpp::Named("offset") = Rcpp::wrap(object.data_.offsets_),
        Rcpp::Named("cross_validation") = cv_res,
        Rcpp::Named("et") = et_res
        );
}

// convert the given control from Rcpp::List to abclass::LinearControl
inline abclass::LinearControl abclass_control(const Rcpp::List& control)
{
    abclass::LinearControl ctrl;
    ctrl.max_iter = control["maxit"];
    ctrl.epsilon = control["epsilon"];
    ctrl.set_intercept(control["intercept"]).
        set_lower_limit(Rcpp::as<Eigen::MatrixXd>(control["lower_limit"])).
        set_upper_limit(Rcpp::as<Eigen::MatrixXd>(control["upper_limit"])).
        reg_path(control["nlambda"],
                 control["lambda_min_ratio"],
                 abclass::rvec2eigen(control["penalty_factor"])).
        reg_lambda(abclass::rvec2eigen(control["lambda"])).
        reg_ridge(control["alpha"],
                  control["lambda_max_alpha_min"]).
        reg_ncv(control["ncv_kappa"]).
        reg_gel(control["gel_tau"]).
        tune_cv(control["nfolds"],
                control["stratified"],
                control["alignment"]).
        tune_et(control["nstages"]);
    return ctrl;
}

// run fit (and optionally cv/et) on a typed object
template <typename T_abclass, typename T_x>
inline void run_fit(T_abclass& obj, const abclass::Data<T_x>& train_data,
                    const Rcpp::List& control)
{
    obj.set_verbose(control["verbose"]);
    if (obj.ctrl_.et_nstages > 0) {
        abclass::et_lambda(obj, train_data);
        if (obj.ctrl_.cv_nfolds > 0) {
            abclass::et_cv_accuracy(obj);
        }
    } else {
        obj.fit(train_data);
        if (obj.ctrl_.cv_nfolds > 0) {
            abclass::cv_lambda(obj);
        }
    }
}

// dispatch over loss_id and penalty_id, call fit, return Rcpp::List
template <typename T_x>
Rcpp::List t_abclass_linear_fit(
    const T_x& x,
    const Eigen::VectorXi& y,
    const Rcpp::List& control
    )
{
    const size_t loss_id { control["loss_id"] };
    const size_t penalty_id { control["penalty_id"] };
    abclass::LinearControl ctrl { abclass_control(control) };
    abclass::Data<T_x> train_data { x, y };
    train_data.set_weights(
        abclass::rvec2eigen(control["weights"]));
    train_data.set_offsets(
        Rcpp::as<Eigen::MatrixXd>(control["offset"]));
    train_data.set_owl_reward(
        abclass::rvec2eigen(control["owl_reward"]));

    // set loss-specific parameters using if constexpr to avoid
    // instantiation errors when the loss type lacks certain setters
    auto set_loss_params = [&](auto& loss_fun,
                               const Rcpp::List& ctrl_list) {
        using LossType = std::decay_t<decltype(loss_fun)>;
        if constexpr (std::is_same_v<LossType, abclass::Boost>) {
            loss_fun.set_inner_min(ctrl_list["boost_umin"]);
        } else if constexpr (std::is_same_v<LossType, abclass::HingeBoost>) {
            loss_fun.set_c(ctrl_list["lum_c"]);
        } else if constexpr (std::is_same_v<LossType, abclass::Lum>) {
            loss_fun.set_ac(ctrl_list["lum_a"], ctrl_list["lum_c"]);
        }
    };

    // helper macro to construct, set loss params, and run fit
#define ABCLASS_FIT_PENALTY(LossType, PenType)                          \
    do {                                                                 \
        abclass::PenType<abclass::LossType, T_x> obj;                   \
        obj.ctrl_ = ctrl;                                                \
        set_loss_params(obj.loss_fun_, control);                        \
        run_fit(obj, train_data, control);                              \
        return get_all_res(obj);                                        \
    } while (false)

    switch (loss_id) {
    case 1:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(Logistic, ElasticNet); }
        case 2: { ABCLASS_FIT_PENALTY(Logistic, SCAD); }
        case 3: { ABCLASS_FIT_PENALTY(Logistic, MCP); }
        case 4: { ABCLASS_FIT_PENALTY(Logistic, GroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(Logistic, GroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(Logistic, GroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(Logistic, CompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(Logistic, GEL); }
        default: break;
        }
        break;
    case 2:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(Boost, ElasticNet); }
        case 2: { ABCLASS_FIT_PENALTY(Boost, SCAD); }
        case 3: { ABCLASS_FIT_PENALTY(Boost, MCP); }
        case 4: { ABCLASS_FIT_PENALTY(Boost, GroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(Boost, GroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(Boost, GroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(Boost, CompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(Boost, GEL); }
        default: break;
        }
        break;
    case 3:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(HingeBoost, ElasticNet); }
        case 2: { ABCLASS_FIT_PENALTY(HingeBoost, SCAD); }
        case 3: { ABCLASS_FIT_PENALTY(HingeBoost, MCP); }
        case 4: { ABCLASS_FIT_PENALTY(HingeBoost, GroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(HingeBoost, GroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(HingeBoost, GroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(HingeBoost, CompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(HingeBoost, GEL); }
        default: break;
        }
        break;
    case 4:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(Lum, ElasticNet); }
        case 2: { ABCLASS_FIT_PENALTY(Lum, SCAD); }
        case 3: { ABCLASS_FIT_PENALTY(Lum, MCP); }
        case 4: { ABCLASS_FIT_PENALTY(Lum, GroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(Lum, GroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(Lum, GroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(Lum, CompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(Lum, GEL); }
        default: break;
        }
        break;
    case 5:
        // "mlogit" loss: former Mlogit class, now an alias for LikeBoost
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(LikeBoost, ElasticNet); }
        case 2: { ABCLASS_FIT_PENALTY(LikeBoost, SCAD); }
        case 3: { ABCLASS_FIT_PENALTY(LikeBoost, MCP); }
        case 4: { ABCLASS_FIT_PENALTY(LikeBoost, GroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(LikeBoost, GroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(LikeBoost, GroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(LikeBoost, CompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(LikeBoost, GEL); }
        default: break;
        }
        break;
    case 6:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(LikeLogistic, ElasticNet); }
        case 2: { ABCLASS_FIT_PENALTY(LikeLogistic, SCAD); }
        case 3: { ABCLASS_FIT_PENALTY(LikeLogistic, MCP); }
        case 4: { ABCLASS_FIT_PENALTY(LikeLogistic, GroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(LikeLogistic, GroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(LikeLogistic, GroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(LikeLogistic, CompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(LikeLogistic, GEL); }
        default: break;
        }
        break;
    case 7:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(LikeBoost, ElasticNet); }
        case 2: { ABCLASS_FIT_PENALTY(LikeBoost, SCAD); }
        case 3: { ABCLASS_FIT_PENALTY(LikeBoost, MCP); }
        case 4: { ABCLASS_FIT_PENALTY(LikeBoost, GroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(LikeBoost, GroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(LikeBoost, GroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(LikeBoost, CompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(LikeBoost, GEL); }
        default: break;
        }
        break;
    default:
        break;
    }

#undef ABCLASS_FIT_PENALTY

    Rcpp::stop("Unsupported loss_id (%zu) or penalty_id (%zu).",
               loss_id, penalty_id);
    return Rcpp::List(); // unreachable
}

// [[Rcpp::export]]
Rcpp::List rcpp_abclass_linear_fit(
    const Eigen::MatrixXd& x,
    const Eigen::VectorXi& y,
    const Rcpp::List& control
    )
{
    return t_abclass_linear_fit<Eigen::MatrixXd>(x, y, control);
}

// [[Rcpp::export]]
Rcpp::List rcpp_abclass_linear_fit_sp(
    const Eigen::SparseMatrix<double>& x,
    const Eigen::VectorXi& y,
    const Rcpp::List& control
    )
{
    return t_abclass_linear_fit<Eigen::SparseMatrix<double>>(x, y, control);
}
