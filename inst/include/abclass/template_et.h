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

#ifndef ABCLASS_TEMPLATE_ET_H
#define ABCLASS_TEMPLATE_ET_H

#include <utility>
#include <RcppArmadillo.h>
#include "CrossValidation.h"
#include "utils.h"

namespace abclass {

    // et-lasso procedure for the entire training set
    //! @param obj An Abclass object
    template <typename T>
    inline void et_lambda(T& obj)
    {
        // record some original data
        const unsigned int p0 { obj.data_.p0_ };
        const unsigned int inter { obj.data_.inter_ };
        const auto x0 { obj.data_.x_ };
        const bool standardize0 { obj.control_.standardize_ };
        const arma::rowvec x_center0 { obj.data_.x_center_ },
            x_scale0 { obj.data_.x_scale_ };
        const arma::mat lower_limit0 { obj.control_.lower_limit_ };
        const arma::mat upper_limit0 { obj.control_.upper_limit_ };
        // 1. set standardize to false to avoid unnessary rescale
        //    as 1) the location/scale do not depend on permutation
        //       2) regardless of standaridze, rescale does nothing
        //          in the following loop
        // 2. set the optional penalty factor from the control object
        obj.set_standardize(false);
        obj.set_penalty_factor();
        const arma::vec gw0 { obj.control_.penalty_factor_ };
        // initialize
        // (0, 1, ...p0 - 1), assuming p0 > 0
        obj.et_vs_ = arma::regspace<arma::uvec>(0, p0 - 1);
        arma::mat active_beta;
        arma::uvec active_idx0;
        // record lambda's
        arma::vec l1_lambda0(obj.control_.et_nstages_),
            l1_lambda1 { l1_lambda0 };
        for (size_t i { 0 }; i < obj.control_.et_nstages_; ++i) {
            // create pseudo-features
            const arma::uvec perm_idx { arma::randperm(obj.data_.n_obs_) };
            auto x_perm { subset_rows(x0, perm_idx) };
            x_perm = arma::join_rows(x0.cols(obj.et_vs_), std::move(x_perm));
            obj.control_.penalty_factor_ = arma::join_cols(
                obj.control_.penalty_factor_.elem(obj.et_vs_), gw0);
            obj.control_.lower_limit_ = arma::join_cols(
                obj.control_.lower_limit_.rows(obj.et_vs_),
                lower_limit0);
            obj.control_.upper_limit_ = arma::join_cols(
                obj.control_.upper_limit_.rows(obj.et_vs_),
                upper_limit0);
            obj.set_x(x_perm);
            obj.et_npermuted_ = p0;
            obj.fit();
            // reset lambda if it was internally set
            if (! obj.control_.custom_lambda_) {
                if (i + 1 < obj.control_.et_nstages_) {
                    // if the last stage has not been done yet
                    double et_lambda_min {
                        std::pow(obj.control_.lambda_min_ratio_, 0.25) *
                        obj.et_l1_lambda1_ /
                        std::max(obj.control_.ridge_alpha_,
                                 obj.control_.lambda_max_alpha_min_)
                    };
                    obj.control_.reg_lambda_min(et_lambda_min);
                } else {
                    // if the last stage has been done
                    obj.control_.reg_lambda_min(-1.0);
                }
            }
            l1_lambda0(i) = obj.et_l1_lambda0_;
            l1_lambda1(i) = obj.et_l1_lambda1_;
            // update active x
            const unsigned int p1_i { obj.data_.p1_ - p0 };
            const unsigned int p0_i { obj.data_.p0_ - p0 };
            const unsigned int et_lambda_idx { obj.coef_.n_slices - 1 };
            active_beta = obj.coef_.slice(et_lambda_idx).head_rows(p1_i);
            arma::uvec pos_beta(p0_i);
            // get the indices of the selected predictors
            for (size_t j { 0 }; j < p0_i; ++j) {
                if (! active_beta.row(inter + j).is_zero()) {
                    pos_beta[j] = 1;
                }
            }
            active_idx0 = arma::find(pos_beta > 0);
            obj.et_vs_ = obj.et_vs_.elem(active_idx0);
            // verbose
            if (obj.control_.verbose_ > 0) {
                Rcpp::Rcout << "[ET] (stage "
                            << i + 1
                            << ") Number of active predictors: "
                            << obj.et_vs_.n_elem
                            << "\n";
            }
            // record loss function
            obj.loss_ = obj.loss_(et_lambda_idx);
            obj.penalty_ = obj.penalty_(et_lambda_idx);
            obj.objective_ = obj.objective_(et_lambda_idx);
        }
        // reset object
        obj.set_standardize(standardize0);
        obj.set_x(std::move(x0));
        obj.set_penalty_factor(std::move(gw0));
        obj.set_coef_lower_limit(std::move(lower_limit0));
        obj.set_coef_upper_limit(std::move(upper_limit0));
        obj.coef_ = arma::cube(obj.data_.p1_, obj.data_.k_ - 1, 1,
                               arma::fill::zeros);
        if (obj.control_.intercept_) {
            obj.coef_.slice(0).rows(obj.et_vs_ + 1) =
                active_beta.rows(active_idx0 + 1);
            // intercept
            obj.coef_.slice(0).row(0) = active_beta.row(0);
        } else {
            obj.coef_.slice(0).rows(obj.et_vs_) = active_beta.rows(active_idx0);
        }
        if (standardize0) {
            // set to the original center and scale
            obj.data_.x_center_ = x_center0;
            obj.data_.x_scale_ = x_scale0;
            obj.force_rescale_coef();
        }
        obj.et_npermuted_ = 0;  // necessary for calling regular fit()
        obj.et_l1_lambda0_vec_ = l1_lambda0;
        obj.et_l1_lambda1_vec_ = l1_lambda1;
    }

    // estimate the prediction accuracy by cross-validation
    // for the model from the et-lasso procedure
    //! @param obj An Abclass object
    //! @param strata optional strata indicator variable for stratified
    //!     sampling in cross validation
    template <typename T>
    inline void et_cv_accuracy(T& obj)
    {
        // default to use y as the strata if stratified is true
        // and strata not specified
        if (obj.control_.cv_stratified_ &&
            obj.control_.cv_strata_.n_elem != obj.data_.n_obs_) {
            obj.control_.cv_strata_ = obj.data_.y_;
        }
        CrossValidation cv_obj {
            obj.data_.n_obs_, obj.control_.cv_nfolds_, obj.control_.cv_strata_
        };
        obj.cv_accuracy_ = arma::zeros(obj.control_.cv_nfolds_);
        for (size_t i { 0 }; i < obj.control_.cv_nfolds_; ++i) {
            auto train_x {
                subset_rows(obj.data_.x_, cv_obj.train_index_.at(i))
            };
            auto test_x {
                subset_rows(obj.data_.x_, cv_obj.test_index_.at(i))
            };
            arma::mat train_offset, test_offset;
            if (obj.control_.has_offset_) {
                train_offset = subset_rows(obj.control_.offset_,
                                           cv_obj.train_index_.at(i));
                test_offset = subset_rows(obj.control_.offset_,
                                          cv_obj.test_index_.at(i));
            }
            arma::uvec train_y {
                obj.data_.y_.rows(cv_obj.train_index_.at(i))
            };
            arma::uvec test_y {
                obj.data_.y_.rows(cv_obj.test_index_.at(i))
            };
            arma::vec train_weight {
                obj.control_.obs_weight_.elem(cv_obj.train_index_.at(i))
            };
            // create a new object
            T new_obj { obj };
            new_obj.set_standardize(false);
            new_obj.set_data(std::move(train_x), std::move(train_y));
            new_obj.enforce_k(obj.data_.k_);
            new_obj.set_weight(std::move(train_weight));
            new_obj.set_offset(std::move(train_offset));
            // alignment: 0 for alignment by fraction
            //            1 for alignment by lambda
            if (! obj.control_.custom_lambda_ &&
                obj.control_.cv_alignment_ == 0) {
                // reset lambda
                new_obj.control_.reg_lambda();
            }
            new_obj.control_.set_verbose(0);
            et_lambda(new_obj);
            obj.cv_accuracy_(i) = new_obj.accuracy(
                    new_obj.coef_.slice(0), test_x, test_y, test_offset);
        }
        obj.cv_accuracy_mean_ = arma::mean(obj.cv_accuracy_);
        obj.cv_accuracy_sd_ = arma::stddev(obj.cv_accuracy_);
    }
}

#endif
