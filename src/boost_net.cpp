#include <RcppArmadillo.h>
#include <abclass.h>
#include "export-helpers.h"

// [[Rcpp::export]]
Rcpp::List rcpp_boost_net(
    const arma::mat& x,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const arma::vec& weight,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int nfolds = 0,
    const bool stratified_cv = true,
    const unsigned int max_iter = 1e5,
    const double rel_tol = 1e-4,
    const bool varying_active_set = true,
    const double inner_min = -3.0,
    const unsigned int verbose = 0
    )
{
    abclass::BoostNet object {
        x, y, inner_min, intercept, standardize, weight
    };
    return abclass_net_fit(object, y,
                           lambda, alpha, nlambda, lambda_min_ratio,
                           nfolds, stratified_cv, max_iter, rel_tol,
                           varying_active_set, verbose);
}
