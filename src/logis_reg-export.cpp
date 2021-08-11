#include <RcppArmadillo.h>
#include <malc.h>


// [[Rcpp::export]]
Rcpp::List rcpp_logistic_reg(
    const arma::mat& x,
    const arma::uvec& y,
    const double lambda,
    const double alpha,
    const arma::mat& penalty_factor,
    const arma::mat& start,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int max_iter = 200,
    const double rel_tol = 1e-3,
    const double pmin = 1e-5,
    const bool early_stop = false,
    const bool verbose = false
    )
{
    Malc::LogisticReg object {
        x, y, intercept, standardize
    };
    object.elastic_net(lambda, alpha, penalty_factor,
                       start, max_iter, rel_tol, pmin, early_stop, verbose);
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("class_prob") = object.prob_mat_,
        Rcpp::Named("en_class_prob") = object.en_prob_mat_,
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") = lambda,
            Rcpp::Named("alpha") = alpha,
            Rcpp::Named("l1_lambda") = object.l1_lambda_,
            Rcpp::Named("l2_lambda") = object.l2_lambda_,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_,
            Rcpp::Named("l1_penalty_factor") = object.l1_penalty_factor_
            )
        );
}


// [[Rcpp::export]]
Rcpp::List rcpp_logistic_path(
    const arma::mat& x,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const arma::mat& penalty_factor,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int max_iter = 200,
    const double rel_tol = 1e-3,
    const double pmin = 1e-5,
    const bool early_stop = false,
    const bool verbose = false
    )
{
    Malc::LogisticReg object {
        x, y, intercept, standardize
    };
    // object.set_offset(offset);
    object.elastic_net_path(lambda, alpha, nlambda, lambda_min_ratio,
                            penalty_factor, max_iter, rel_tol,
                            pmin, early_stop, verbose);
    Rcpp::NumericVector lambda_vec { Malc::arma2rvec(object.lambda_path_) };
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_path_,
        Rcpp::Named("class_prob") = object.prob_path_,
        Rcpp::Named("en_class_prob") = object.en_prob_path_,
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") = lambda_vec,
            Rcpp::Named("alpha") = alpha,
            Rcpp::Named("l1_lambda") = alpha * lambda_vec,
            Rcpp::Named("l2_lambda") = 0.5 * (1 - alpha) * lambda_vec,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_,
            Rcpp::Named("l1_penalty_factor") = object.l1_penalty_factor_
            )
        );
}
