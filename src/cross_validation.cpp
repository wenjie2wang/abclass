#include <RcppArmadillo.h>
#include <abclass.h>

// [[Rcpp::export]]
Rcpp::List cv_samples(const unsigned int nobs,
                      const unsigned int nfolds,
                      const arma::uvec& strata)
{
    abclass::CrossValidation cv_obj {
        nobs, nfolds, strata
    };
    Rcpp::List train_list, valid_list;
    for (size_t i {0}; i < nfolds; ++i) {
        train_list.push_back(abclass::arma2rvec(cv_obj.train_index_.at(i)));
        valid_list.push_back(abclass::arma2rvec(cv_obj.test_index_.at(i)));
    }
    return Rcpp::List::create(
        Rcpp::Named("train_index") = train_list,
        Rcpp::Named("valid_index") = valid_list
        );
}
