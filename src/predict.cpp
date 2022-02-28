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
arma::uvec rcpp_logistic_predict_y(const arma::mat& beta,
                                   const arma::mat& x)
{
    return arma::index_max(rcpp_logistic_predict_prob(beta, x), 1);
}

// [[Rcpp::export]]
arma::mat rcpp_boost_predict_prob(const arma::mat& beta,
                                  const arma::mat& x,
                                  const double inner_min)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::BoostNet object { k };
    object.set_inner_min(inner_min);
    return object.predict_prob(beta, x);
}

// [[Rcpp::export]]
arma::uvec rcpp_boost_predict_y(const arma::mat& beta,
                                const arma::mat& x,
                                const double inner_min)
{
    return arma::index_max(rcpp_boost_predict_prob(beta, x, inner_min), 1);
}

// [[Rcpp::export]]
arma::mat rcpp_hinge_boost_predict_prob(const arma::mat& beta,
                                        const arma::mat& x,
                                        const double lum_c)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::HingeBoostNet object { k };
    object.set_lum_c(lum_c);
    return object.predict_prob(beta, x);
}

// [[Rcpp::export]]
arma::uvec rcpp_hinge_boost_predict_y(const arma::mat& beta,
                                      const arma::mat& x,
                                      const double lum_c)
{
    return arma::index_max(rcpp_hinge_boost_predict_prob(beta, x, lum_c), 1);
}

// [[Rcpp::export]]
arma::mat rcpp_lum_predict_prob(const arma::mat& beta,
                                const arma::mat& x,
                                const double lum_a,
                                const double lum_c)
{
    const unsigned int k { beta.n_cols + 1 };
    abclass::LumNet object { k };
    object.set_lum_parameters(lum_a, lum_c);
    return object.predict_prob(beta, x);
}

// [[Rcpp::export]]
arma::uvec rcpp_lum_predict_y(const arma::mat& beta,
                              const arma::mat& x,
                              const double lum_a,
                              const double lum_c)
{
    return arma::index_max(rcpp_lum_predict_prob(beta, x, lum_a, lum_c), 1);
}
