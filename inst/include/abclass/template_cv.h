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

#ifndef ABCLASS_TEMPLATE_CV_H
#define ABCLASS_TEMPLATE_CV_H

#include <utility>
#include <RcppArmadillo.h>
#include "AbclassNet.h"
#include "AbclassGroupLasso.h"
#include "AbclassGroupSCAD.h"
#include "AbclassGroupMCP.h"
#include "CrossValidation.h"
#include "utils.h"

namespace abclass {

    // cross-validation method for Abclass objects
    //! @param alignment 0 for alignment by fraction, 1 for alignment by lambda
    template <typename T>
    inline void cv_lambda(T& obj, const arma::uvec strata = arma::uvec())
    {
        CrossValidation cv_obj {
            obj.n_obs_, obj.control_.cv_nfolds_, strata
        };
        size_t ntune { obj.control_.lambda_.n_elem };
        if (ntune == 0) {
            ntune = obj.control_.nlambda_;
        }
        obj.cv_accuracy_ = arma::zeros(ntune, obj.control_.cv_nfolds_);
        // model fits
        for (size_t i { 0 }; i < obj.control_.cv_nfolds_; ++i) {
            auto train_x { subset_rows(obj.x_, cv_obj.train_index_.at(i)) };
            arma::uvec train_y { obj.y_.rows(cv_obj.train_index_.at(i)) };
            arma::vec train_weight {
                obj.control_.obs_weight_.elem(cv_obj.train_index_.at(i))
            };
            auto test_x { subset_rows(obj.x_, cv_obj.test_index_.at(i)) };
            arma::uvec test_y { obj.y_.rows(cv_obj.test_index_.at(i)) };
            // create a new object
            T new_obj { obj };
            new_obj.set_standardize(false);
            new_obj.set_data(std::move(train_x),
                             std::move(train_y))->set_k(obj.k_);
            new_obj.set_weight(std::move(train_weight));
            if (! obj.custom_lambda_ && obj.control_.cv_alignment_ == 0) {
                // reset lambda
                new_obj.control_.reg_path(arma::vec());
            }
            new_obj.control_.set_verbose(0);
            new_obj.fit();
            for (size_t l { 0 }; l < ntune; ++l) {
                obj.cv_accuracy_(l, i) = new_obj.accuracy(
                    new_obj.coef_.slice(l), test_x, test_y);
            }
        }
        obj.cv_accuracy_mean_ = mat2vec(arma::mean(obj.cv_accuracy_, 1));
        obj.cv_accuracy_sd_ = mat2vec(arma::stddev(obj.cv_accuracy_, 0, 1));
    }

}

#endif
