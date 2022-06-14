#include <RcppArmadillo.h>
#include <abclass.h>

// for AbclassNet objects
template <typename T>
Rcpp::List abclass_net_fit(T& object)
{
    object.fit();
    if (object.control_.cv_nfolds_ > 0) {
        arma::uvec strata;
        if (object.control_.cv_stratified_) {
            strata = object.y_;
        }
        abclass::cv_lambda(object,
                           object.control_.cv_nfolds_,
                           strata,
                           object.control_.cv_alignment_);
    }
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("weight") = abclass::arma2rvec(object.control_.obs_weight_),
        Rcpp::Named("cross_validation") = Rcpp::List::create(
            Rcpp::Named("nfolds") = object.control_.cv_nfolds_,
            Rcpp::Named("stratified") = object.control_.cv_stratified_,
            Rcpp::Named("alignment") = object.control_.cv_alignment_,
            Rcpp::Named("cv_accuracy") = object.cv_accuracy_,
            Rcpp::Named("cv_accuracy_mean") =
            abclass::arma2rvec(object.cv_accuracy_mean_),
            Rcpp::Named("cv_accuracy_sd") =
            abclass::arma2rvec(object.cv_accuracy_sd_)
            ),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") =
            abclass::arma2rvec(object.control_.lambda_),
            Rcpp::Named("alpha") = object.control_.alpha_,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_
            )
        );
}

// for AbclassGroupLasso objects
template <typename T>
Rcpp::List abclass_group_lasso_fit(T& object)
{
    object.fit();
    if (object.control_.cv_nfolds_ > 0) {
        arma::uvec strata;
        if (object.control_.cv_stratified_) {
            strata = object.y_;
        }
        abclass::cv_lambda(object,
                           object.control_.cv_nfolds_,
                           strata,
                           object.control_.cv_alignment_);
    }
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("weight") = abclass::arma2rvec(object.control_.obs_weight_),
        Rcpp::Named("cross_validation") = Rcpp::List::create(
            Rcpp::Named("nfolds") = object.control_.cv_nfolds_,
            Rcpp::Named("stratified") = object.control_.cv_stratified_,
            Rcpp::Named("alignment") = object.control_.cv_alignment_,
            Rcpp::Named("cv_accuracy") = object.cv_accuracy_,
            Rcpp::Named("cv_accuracy_mean") =
            abclass::arma2rvec(object.cv_accuracy_mean_),
            Rcpp::Named("cv_accuracy_sd") =
            abclass::arma2rvec(object.cv_accuracy_sd_)
            ),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") =
            abclass::arma2rvec(object.control_.lambda_),
            Rcpp::Named("group_weight") =
            abclass::arma2rvec(object.control_.group_weight_),
            Rcpp::Named("lambda_max") = object.lambda_max_
            )
        );
}

// for AbclassGroupSCAD/AbclassGroupMCP objects
template <typename T>
Rcpp::List abclass_group_ncv_fit(T& object)
{
    object.fit();
    if (object.control_.cv_nfolds_ > 0) {
        arma::uvec strata;
        if (object.control_.cv_stratified_) {
            strata = object.y_;
        }
        abclass::cv_lambda(object,
                           object.control_.cv_nfolds_,
                           strata,
                           object.control_.cv_alignment_);
    }
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("weight") = abclass::arma2rvec(object.control_.obs_weight_),
        Rcpp::Named("cross_validation") = Rcpp::List::create(
            Rcpp::Named("nfolds") = object.control_.cv_nfolds_,
            Rcpp::Named("stratified") = object.control_.cv_stratified_,
            Rcpp::Named("alignment") = object.control_.cv_alignment_,
            Rcpp::Named("cv_accuracy") = object.cv_accuracy_,
            Rcpp::Named("cv_accuracy_mean") =
            abclass::arma2rvec(object.cv_accuracy_mean_),
            Rcpp::Named("cv_accuracy_sd") =
            abclass::arma2rvec(object.cv_accuracy_sd_)
            ),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") =
            abclass::arma2rvec(object.control_.lambda_),
            Rcpp::Named("group_weight") =
            abclass::arma2rvec(object.control_.group_weight_),
            Rcpp::Named("dgamma") = object.control_.dgamma_,
            Rcpp::Named("gamma") = object.gamma_,
            Rcpp::Named("lambda_max") = object.lambda_max_
            )
        );
}
