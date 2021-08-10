#include <RcppArmadillo.h>
#include <malc.h>


// [[Rcpp::export]]
Rcpp::List rcpp_logis_reg(
    const arma::mat& x,
    const arma::uvec& y,
    const double lambda,
    const double alpha,
    const bool intercept,
    const bool standardize,
    const arma::mat& penalty_factor,
    const arma::mat& offset,
    const arma::mat& start,
    const unsigned int max_iter,
    const double rel_tol,
    const double pmin,
    const bool early_stop,
    const bool verbose
    )
{
    Malc::LogisticReg object {
        x, y, intercept, standardize
    };
    double l1_lambda { lambda * alpha };
    double l2_lambda { lambda * (1 - alpha) * 0.5 };
    // object.set_offset(offset);
    object.elastic_net(l1_lambda, l2_lambda, penalty_factor,
                       start, max_iter, rel_tol, pmin, early_stop, verbose);
    return Rcpp::List::create(
        Rcpp::Named("coef") = object.coef_,
        Rcpp::Named("prob") = object.prob_mat_
        );
}
