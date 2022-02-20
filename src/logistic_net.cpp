#include <RcppArmadillo.h>
#include <abclass.h>

// [[Rcpp::export]]
Rcpp::List rcpp_logistic_net(
    const arma::mat& x,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const arma::vec& weight,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int max_iter = 1e4,
    const double rel_tol = 1e-4,
    const bool varying_active_set = true,
    const unsigned int verbose = 0
    )
{
    abclass::LogisticNet object {
        x, y, intercept, standardize, weight
    };
    object.fit(lambda, alpha, nlambda, lambda_min_ratio,
               max_iter, rel_tol, varying_active_set, verbose);
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("weight") = abclass::arma2rvec(object.get_weight()),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") = lambda,
            Rcpp::Named("alpha") = alpha,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_
            )
        );
}
