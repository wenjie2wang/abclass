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
             control["lambda_weight"])->
        set_offset(control["offset"])->
        reg_path(control["nlambda"],
                 control["lambda_min_ratio"],
                 control["varying_active_set"])->
        reg_path(control["lambda"])->
        reg_net(control["alpha"])->
        tune_cv(control["nfolds"])->
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
    if (object.abc_.control_.cv_nfolds_ > 0) {
        cv_res = Rcpp::List::create(
            Rcpp::Named("cv_recall") = object.cv_abrank_recall()
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
    std::vector<arma::mat> xs;
    std::vector<arma::vec> ys;
    const size_t nquery { qid.max() + 1 };
    for (size_t i {0}; i < nquery; ++i) {
        arma::uvec is_i { arma::find(qid == i) };
        arma::mat x_i { x.rows(is_i) };
        arma::vec y_i { y.elem(is_i) };
        xs.push_back(x_i);
        ys.push_back(y_i);
    }
    const size_t loss_id { control["loss_id"] };
    abclass::Control ctrl { abrank_control(control) };
    switch (loss_id) {
        case 1: {
            abclass::LogisticRank<arma::mat> object { xs, ys, ctrl };
            return template_abrank_fit(object);
        }
        case 2: {
            abclass::BoostRank<arma::mat> object { xs, ys, ctrl };
            object.abc_.loss_.set_inner_min(control["boost_umin"]);
            return template_abrank_fit(object);
        }
        case 3: {
            abclass::HingeBoostRank<arma::mat> object { xs, ys, ctrl };
            object.abc_.loss_.set_c(control["lum_c"]);
            return template_abrank_fit(object);
        }
        case 4: {
            abclass::LumRank<arma::mat> object { xs, ys, ctrl };
            object.abc_.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_abrank_fit(object);
        }
        default:
            break;
    }
    return Rcpp::List();
}
