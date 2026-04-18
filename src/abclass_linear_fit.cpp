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
template <typename T_x>
inline Rcpp::List get_et_res(const abclass::Output<T_x>& object)
{
    return Rcpp::List::create(
        Rcpp::Named("nstages") = object.control_.et_nstages_,
        Rcpp::Named("selected") = Rcpp::wrap(object.et_vs_),
        Rcpp::Named("l1_lambda0") =
            abclass::eigen2rvec(object.et_l1_lambda0_vec_),
        Rcpp::Named("l1_lambda1") =
            abclass::eigen2rvec(object.et_l1_lambda1_vec_));
}

// returns for cv procedure
template <typename T_x>
inline Rcpp::List get_cv_res(const abclass::Output<T_x>& object)
{
    return Rcpp::List::create(
        Rcpp::Named("nfolds") = object.control_.cv_nfolds_,
        Rcpp::Named("stratified") = object.control_.cv_stratified_,
        Rcpp::Named("alignment") = object.control_.cv_alignment_,
        Rcpp::Named("cv_accuracy") = Rcpp::wrap(object.cv_accuracy_),
        Rcpp::Named("cv_accuracy_mean") =
            abclass::eigen2rvec(object.cv_accuracy_mean_),
        Rcpp::Named("cv_accuracy_sd") =
            abclass::eigen2rvec(object.cv_accuracy_sd_));
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
        Rcpp::Named("coefficients") = coef_to_array(object.coef_),
        Rcpp::Named("optimization") = Rcpp::List::create(
            Rcpp::Named("loss") = abclass::eigen2rvec(object.loss_),
            Rcpp::Named("penalty") = abclass::eigen2rvec(object.penalty_),
            Rcpp::Named("objective") = abclass::eigen2rvec(object.objective_),
            Rcpp::Named("n_iterations") = object.n_iter_
            ),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") =
            abclass::eigen2rvec(object.control_.lambda_),
            Rcpp::Named("alpha") = object.control_.ridge_alpha_,
            Rcpp::Named("penalty_factor") =
            abclass::eigen2rvec(object.control_.penalty_factor_),
            Rcpp::Named("lambda_max") = object.lambda_max_,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_,
            Rcpp::Named("ncv_kappa") = object.control_.ncv_kappa_,
            Rcpp::Named("ncv_gamma") = object.control_.ncv_gamma_,
            Rcpp::Named("gel_tau") = object.control_.gel_tau_,
            Rcpp::Named("mellowmax_omega") = object.control_.mellowmax_omega_
            ),
        Rcpp::Named("weights") =
        abclass::eigen2rvec(object.data_.obs_weights_),
        Rcpp::Named("offset") = Rcpp::wrap(object.data_.offset_),
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
    ctrl.set_lower_limit(Rcpp::as<Eigen::MatrixXd>(control["lower_limit"])).
        set_upper_limit(Rcpp::as<Eigen::MatrixXd>(control["upper_limit"])).
        reg_path(control["nlambda"],
                 control["lambda_min_ratio"],
                 abclass::rvec2eigen(control["penalty_factor"]),
                 control["varying_active_set"],
                 control["adjust_mm"]).
        reg_lambda(abclass::rvec2eigen(control["lambda"])).
        reg_ridge(control["alpha"],
                  control["lambda_max_alpha_min"]).
        reg_ncv(control["ncv_kappa"]).
        reg_gel(control["gel_tau"]).
        reg_mellowmax(control["mellowmax_omega"]).
        tune_cv(control["nfolds"],
                control["stratified"],
                control["alignment"]).
        tune_et(control["nstages"]);
    return ctrl;
}

// run fit (and optionally cv/et) on a typed object, return output
template <typename T_abclass>
auto run_fit(T_abclass& obj) -> decltype(obj.output_)
{
    if (obj.control_.et_nstages_ > 0) {
        abclass::et_lambda(obj);
        if (obj.control_.cv_nfolds_ > 0) {
            abclass::et_cv_accuracy(obj);
        }
    } else {
        obj.fit();
        if (obj.control_.cv_nfolds_ > 0) {
            abclass::cv_lambda(obj);
        }
    }
    obj.output_.control_ = obj.control_;
    return obj.output_;
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
    abclass::Control ctrl { abclass_control(control) };
    abclass::Data<T_x> train_data { x, y };
    train_data.set_obs_weights(
        abclass::rvec2eigen(control["weights"]));
    train_data.set_offset(
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
        abclass::PenType<abclass::LossType, T_x> obj { train_data, ctrl }; \
        set_loss_params(obj.loss_fun_, control);                        \
        return get_all_res(run_fit(obj));                                \
    } while (false)

    switch (loss_id) {
    case 1:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(Logistic, AbclassNet); }
        case 2: { ABCLASS_FIT_PENALTY(Logistic, AbclassSCAD); }
        case 3: { ABCLASS_FIT_PENALTY(Logistic, AbclassMCP); }
        case 4: { ABCLASS_FIT_PENALTY(Logistic, AbclassGroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(Logistic, AbclassGroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(Logistic, AbclassGroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(Logistic, AbclassCompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(Logistic, AbclassGEL); }
        default: break;
        }
        break;
    case 2:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(Boost, AbclassNet); }
        case 2: { ABCLASS_FIT_PENALTY(Boost, AbclassSCAD); }
        case 3: { ABCLASS_FIT_PENALTY(Boost, AbclassMCP); }
        case 4: { ABCLASS_FIT_PENALTY(Boost, AbclassGroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(Boost, AbclassGroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(Boost, AbclassGroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(Boost, AbclassCompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(Boost, AbclassGEL); }
        default: break;
        }
        break;
    case 3:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(HingeBoost, AbclassNet); }
        case 2: { ABCLASS_FIT_PENALTY(HingeBoost, AbclassSCAD); }
        case 3: { ABCLASS_FIT_PENALTY(HingeBoost, AbclassMCP); }
        case 4: { ABCLASS_FIT_PENALTY(HingeBoost, AbclassGroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(HingeBoost, AbclassGroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(HingeBoost, AbclassGroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(HingeBoost, AbclassCompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(HingeBoost, AbclassGEL); }
        default: break;
        }
        break;
    case 4:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(Lum, AbclassNet); }
        case 2: { ABCLASS_FIT_PENALTY(Lum, AbclassSCAD); }
        case 3: { ABCLASS_FIT_PENALTY(Lum, AbclassMCP); }
        case 4: { ABCLASS_FIT_PENALTY(Lum, AbclassGroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(Lum, AbclassGroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(Lum, AbclassGroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(Lum, AbclassCompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(Lum, AbclassGEL); }
        default: break;
        }
        break;
    case 5:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(Mlogit, AbclassNet); }
        case 2: { ABCLASS_FIT_PENALTY(Mlogit, AbclassSCAD); }
        case 3: { ABCLASS_FIT_PENALTY(Mlogit, AbclassMCP); }
        case 4: { ABCLASS_FIT_PENALTY(Mlogit, AbclassGroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(Mlogit, AbclassGroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(Mlogit, AbclassGroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(Mlogit, AbclassCompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(Mlogit, AbclassGEL); }
        default: break;
        }
        break;
    case 6:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(LikeLogistic, AbclassNet); }
        case 2: { ABCLASS_FIT_PENALTY(LikeLogistic, AbclassSCAD); }
        case 3: { ABCLASS_FIT_PENALTY(LikeLogistic, AbclassMCP); }
        case 4: { ABCLASS_FIT_PENALTY(LikeLogistic, AbclassGroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(LikeLogistic, AbclassGroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(LikeLogistic, AbclassGroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(LikeLogistic, AbclassCompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(LikeLogistic, AbclassGEL); }
        default: break;
        }
        break;
    case 7:
        switch (penalty_id) {
        case 1: { ABCLASS_FIT_PENALTY(LikeBoost, AbclassNet); }
        case 2: { ABCLASS_FIT_PENALTY(LikeBoost, AbclassSCAD); }
        case 3: { ABCLASS_FIT_PENALTY(LikeBoost, AbclassMCP); }
        case 4: { ABCLASS_FIT_PENALTY(LikeBoost, AbclassGroupLasso); }
        case 5: { ABCLASS_FIT_PENALTY(LikeBoost, AbclassGroupSCAD); }
        case 6: { ABCLASS_FIT_PENALTY(LikeBoost, AbclassGroupMCP); }
        case 7: { ABCLASS_FIT_PENALTY(LikeBoost, AbclassCompMCP); }
        case 8: { ABCLASS_FIT_PENALTY(LikeBoost, AbclassGEL); }
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
