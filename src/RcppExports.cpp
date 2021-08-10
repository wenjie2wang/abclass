// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/malc.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_logis_reg
Rcpp::List rcpp_logis_reg(const arma::mat& x, const arma::uvec& y, const double lambda, const double alpha, const bool intercept, const bool standardize, const arma::mat& penalty_factor, const arma::mat& offset, const arma::mat& start, const unsigned int max_iter, const double rel_tol, const double pmin, const bool early_stop, const bool verbose);
RcppExport SEXP _malc_rcpp_logis_reg(SEXP xSEXP, SEXP ySEXP, SEXP lambdaSEXP, SEXP alphaSEXP, SEXP interceptSEXP, SEXP standardizeSEXP, SEXP penalty_factorSEXP, SEXP offsetSEXP, SEXP startSEXP, SEXP max_iterSEXP, SEXP rel_tolSEXP, SEXP pminSEXP, SEXP early_stopSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const bool >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< const bool >::type standardize(standardizeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type penalty_factor(penalty_factorSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type offset(offsetSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type start(startSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const double >::type rel_tol(rel_tolSEXP);
    Rcpp::traits::input_parameter< const double >::type pmin(pminSEXP);
    Rcpp::traits::input_parameter< const bool >::type early_stop(early_stopSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_logis_reg(x, y, lambda, alpha, intercept, standardize, penalty_factor, offset, start, max_iter, rel_tol, pmin, early_stop, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_malc_rcpp_logis_reg", (DL_FUNC) &_malc_rcpp_logis_reg, 14},
    {NULL, NULL, 0}
};

RcppExport void R_init_malc(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
