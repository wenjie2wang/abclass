// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/abclass.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_logistic_net
Rcpp::List rcpp_logistic_net(const arma::mat& x, const arma::uvec& y, const double lambda, const double alpha, const arma::mat& start, const arma::vec& weight, const bool intercept, const bool standardize, const unsigned int max_iter, const double rel_tol, const double pmin, const bool verbose);
RcppExport SEXP _abclass_rcpp_logistic_net(SEXP xSEXP, SEXP ySEXP, SEXP lambdaSEXP, SEXP alphaSEXP, SEXP startSEXP, SEXP weightSEXP, SEXP interceptSEXP, SEXP standardizeSEXP, SEXP max_iterSEXP, SEXP rel_tolSEXP, SEXP pminSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type start(startSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< const bool >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< const bool >::type standardize(standardizeSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const double >::type rel_tol(rel_tolSEXP);
    Rcpp::traits::input_parameter< const double >::type pmin(pminSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_logistic_net(x, y, lambda, alpha, start, weight, intercept, standardize, max_iter, rel_tol, pmin, verbose));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_logistic_net_path
Rcpp::List rcpp_logistic_net_path(const arma::mat& x, const arma::uvec& y, const arma::vec& lambda, const double alpha, const unsigned int nlambda, const double lambda_min_ratio, const arma::vec& weight, const bool intercept, const bool standardize, const unsigned int max_iter, const double rel_tol, const double pmin, const bool verbose);
RcppExport SEXP _abclass_rcpp_logistic_net_path(SEXP xSEXP, SEXP ySEXP, SEXP lambdaSEXP, SEXP alphaSEXP, SEXP nlambdaSEXP, SEXP lambda_min_ratioSEXP, SEXP weightSEXP, SEXP interceptSEXP, SEXP standardizeSEXP, SEXP max_iterSEXP, SEXP rel_tolSEXP, SEXP pminSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type nlambda(nlambdaSEXP);
    Rcpp::traits::input_parameter< const double >::type lambda_min_ratio(lambda_min_ratioSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< const bool >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< const bool >::type standardize(standardizeSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const double >::type rel_tol(rel_tolSEXP);
    Rcpp::traits::input_parameter< const double >::type pmin(pminSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_logistic_net_path(x, y, lambda, alpha, nlambda, lambda_min_ratio, weight, intercept, standardize, max_iter, rel_tol, pmin, verbose));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_prob_mat
arma::mat rcpp_prob_mat(const arma::mat& beta, const arma::mat& x);
RcppExport SEXP _abclass_rcpp_prob_mat(SEXP betaSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_prob_mat(beta, x));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_predict_cat
arma::uvec rcpp_predict_cat(const arma::mat& prob_mat);
RcppExport SEXP _abclass_rcpp_predict_cat(SEXP prob_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type prob_mat(prob_matSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_predict_cat(prob_mat));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_accuracy
Rcpp::List rcpp_accuracy(const arma::mat& new_x, const arma::uvec& new_y, const arma::mat& beta);
RcppExport SEXP _abclass_rcpp_accuracy(SEXP new_xSEXP, SEXP new_ySEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type new_x(new_xSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type new_y(new_ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_accuracy(new_x, new_y, beta));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_abclass_rcpp_logistic_net", (DL_FUNC) &_abclass_rcpp_logistic_net, 12},
    {"_abclass_rcpp_logistic_net_path", (DL_FUNC) &_abclass_rcpp_logistic_net_path, 13},
    {"_abclass_rcpp_prob_mat", (DL_FUNC) &_abclass_rcpp_prob_mat, 2},
    {"_abclass_rcpp_predict_cat", (DL_FUNC) &_abclass_rcpp_predict_cat, 1},
    {"_abclass_rcpp_accuracy", (DL_FUNC) &_abclass_rcpp_accuracy, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_abclass(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
