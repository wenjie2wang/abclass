//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2025 Eli Lilly and Company
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

#include <RcppArmadillo.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "CrossValidation.h"
#include "utils.h"

namespace abclass
{

    // cross-validation method
    //! @param obj An Abclass object
    template <typename T> inline void cv_lambda(T& obj)
    {
        // require data
        if (obj.empty_data()) {
            throw std::invalid_argument("Data required for cv_lambda()");
        }
        // default to use y as the strata if stratified is true
        // and strata not specified
        if (obj.control_.cv_stratified_ &&
            obj.control_.cv_strata_.n_elem != obj.p_data_->n_obs_) {
            obj.control_.cv_strata_ = obj.p_data_->y_;
        }
        CrossValidation cv_obj{obj.p_data_->n_obs_, obj.control_.cv_nfolds_,
                               obj.control_.cv_strata_};
        size_t ntune{obj.control_.lambda_.n_elem};
        if (ntune == 0) {
            ntune = obj.control_.nlambda_;
        }
        obj.output_.cv_accuracy_ = arma::zeros(ntune, obj.control_.cv_nfolds_);
        // using DataType = std::remove_const_t<decltype(*obj.p_data_)>;
        // model fits
        for (size_t i{0}; i < obj.control_.cv_nfolds_; ++i) {
            auto train_x{
                subset_rows(obj.p_data_->x_, cv_obj.train_index_.at(i))};
            auto test_x{subset_rows(obj.p_data_->x_, cv_obj.test_index_.at(i))};
            arma::mat train_offset, test_offset;
            if (obj.p_data_->has_offset_) {
                train_offset = subset_rows(obj.p_data_->offset_,
                                           cv_obj.train_index_.at(i));
                test_offset =
                    subset_rows(obj.p_data_->offset_, cv_obj.test_index_.at(i));
            }
            arma::uvec train_y{obj.p_data_->y_.rows(cv_obj.train_index_.at(i))};
            arma::uvec test_y{obj.p_data_->y_.rows(cv_obj.test_index_.at(i))};
            arma::vec train_weight{
                obj.p_data_->obs_weights_.elem(cv_obj.train_index_.at(i))};
            // create a new object
            // DataType new_data{obj.p_data_->k_, obj.p_data_->intercept_, false};
            auto new_data{*obj.p_data_};
            new_data.set_data(std::move(train_x), std::move(train_y));
            new_data.set_obs_weights(std::move(train_weight));
            new_data.set_offset(std::move(train_offset));
            T new_obj{new_data, obj.control_};
            // alignment: 0 for alignment by fraction
            //            1 for alignment by lambda
            if (!obj.control_.custom_lambda_ &&
                obj.control_.cv_alignment_ == 0) {
                // reset lambda and set custom lambda to false
                new_obj.control_.reg_lambda();
            }
            new_obj.control_.set_verbose(0);
            new_obj.fit();
            for (size_t l{0}; l < ntune; ++l) {
                obj.output_.cv_accuracy_(l, i) =
                    new_obj.accuracy(new_obj.output_.coef_.slice(l), test_x,
                                     test_y, test_offset);
            }
        }
        obj.output_.cv_accuracy_mean_ =
            mat2vec(arma::mean(obj.output_.cv_accuracy_, 1));
        obj.output_.cv_accuracy_sd_ =
            mat2vec(arma::stddev(obj.output_.cv_accuracy_, 0, 1));
    }

} // namespace abclass

#endif
