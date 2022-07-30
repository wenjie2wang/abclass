//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2022 Eli Lilly and Company
//
// This file is part of the R package abclass.
//
// The R package abclass is free software: You can redistribute it and/or
// modify it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or any later
// version (at your option). See the GNU General Public License at
// <https://www.gnu.org/licenses/> for details.
//
// The R package abclass is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//

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
