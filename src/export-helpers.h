#include <RcppArmadillo.h>
#include <abclass.h>

// for AbclassNet objects
template <typename T>
Rcpp::List abclass_net_fit(
    T& object,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const unsigned int nfolds = 0,
    const bool stratified_cv = true,
    const unsigned int max_iter = 1e4,
    const double rel_tol = 1e-4,
    const bool varying_active_set = true,
    const unsigned int verbose = 0
    )
{
    object.fit(lambda, alpha, nlambda, lambda_min_ratio,
               max_iter, rel_tol, varying_active_set, verbose);
    Rcpp::NumericVector lambda_vec { abclass::arma2rvec(object.lambda_) };
    if (nfolds > 0) {
        arma::uvec strata;
        if (stratified_cv) {
            strata = y;
        }
        abclass::abclass_net_cv(object, nfolds, strata);
    }
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("weight") = abclass::arma2rvec(object.obs_weight_),
        Rcpp::Named("cross_validation") = Rcpp::List::create(
            Rcpp::Named("nfolds") = nfolds,
            Rcpp::Named("stratified") = stratified_cv,
            Rcpp::Named("cv_accuracy") = object.cv_accuracy_,
            Rcpp::Named("cv_accuracy_mean") =
            abclass::arma2rvec(object.cv_accuracy_mean_),
            Rcpp::Named("cv_accuracy_sd") =
            abclass::arma2rvec(object.cv_accuracy_sd_)
            ),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") = abclass::arma2rvec(lambda_vec),
            Rcpp::Named("alpha") = alpha,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_
            )
        );
}
