#include <RcppArmadillo.h>
#include <abclass.h>

// for Abclass objects
template <typename T>
Rcpp::List template_fit(T& object, const bool main_fit)
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
        if (! main_fit) {
            return cv_res;
        }
    }
    // else
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
