#include <cmath>
#include <RcppArmadillo.h>
#include <abclass.h>

// [[Rcpp::export]]
arma::mat rcpp_logistic_predict_prob(const arma::mat& beta,
                                     const arma::mat& x)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::LogisticNet object { k };
    return object.predict_prob(beta, x);
}

// [[Rcpp::export]]
arma::uvec rcpp_logistic_predict_y0(const arma::mat& prob_mat)
{
    return arma::index_max(prob_mat, 1);
}

// [[Rcpp::export]]
arma::uvec rcpp_logistic_predict_y(const arma::mat& beta,
                                   const arma::mat& x)
{
    return rcpp_logistic_predict_y0(rcpp_logistic_predict_prob(beta, x));
}
