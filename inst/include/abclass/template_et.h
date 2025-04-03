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

#include <RcppArmadillo.h>
#include <type_traits>
#include <utility>

#include "Control.h"
#include "CrossValidation.h"
#include "utils.h"

namespace abclass
{

    // et-lasso procedure for the entire training set
    //! @param obj An AbclassCD object
    template <typename T> inline void et_lambda(T& obj)
    {
        // require data
        if (obj.empty_data()) {
            throw std::invalid_argument("Data required for et_lambda()");
        }
        // record some original data
        const unsigned int p0{obj.p_data_->p0_};
        const unsigned int inter{obj.p_data_->inter_};
        const auto x0{obj.p_data_->x_};
        const bool standardize0{obj.p_data_->standardize_};
        obj.pre_fit();
        const arma::vec gw0{obj.control_.penalty_factor_};
        const arma::mat lower_limit0{obj.control_.lower_limit_};
        const arma::mat upper_limit0{obj.control_.upper_limit_};
        // using DataType = std::remove_const_t<decltype(*obj.p_data_)>;
        // initialize
        // (0, 1, ...p0 - 1), assuming p0 > 0
        obj.output_.et_vs_ = arma::regspace<arma::uvec>(0, p0 - 1);
        arma::mat active_beta;
        arma::uvec active_idx0;
        double et_lambda_min{obj.control_.lambda_min_};
        // record lambda's
        arma::vec l1_lambda0(obj.control_.et_nstages_), l1_lambda1{l1_lambda0};
        for (size_t i{0}; i < obj.control_.et_nstages_; ++i) {
            // create pseudo-features
            const arma::uvec perm_idx{arma::randperm(obj.p_data_->n_obs_)};
            auto x_perm{subset_rows(x0, perm_idx)};
            x_perm =
                arma::join_rows(x0.cols(obj.output_.et_vs_), std::move(x_perm));
            // create a new object
            // DataType new_data{*obj.p_data_};
            auto new_data{*obj.p_data_};//
            // set standardize to false to avoid unnessary rescale
            //    as the location/scale do not depend on permutation
            new_data.set_standardize(false);
            new_data.set_x(x_perm);
            new_data.et_npermuted_ = p0;
            Control new_ctrl{obj.control_};
            new_ctrl.reg_lambda_min(et_lambda_min);
            new_ctrl.penalty_factor_ = arma::join_cols(
                obj.control_.penalty_factor_.elem(obj.output_.et_vs_), gw0);
            new_ctrl.lower_limit_ = arma::join_cols(
                obj.control_.lower_limit_.rows(obj.output_.et_vs_),
                lower_limit0);
            new_ctrl.upper_limit_ = arma::join_cols(
                obj.control_.upper_limit_.rows(obj.output_.et_vs_),
                upper_limit0);
            T new_obj{new_data, new_ctrl};
            new_obj.fit();
            // reset lambda if it was internally set
            if (!new_obj.control_.custom_lambda_) {
                // if the last stage has not been done yet
                if (i + 1 < new_obj.control_.et_nstages_) {
                    // the smallest lambda
                    double min_lambda_i{new_obj.control_.lambda_(
                        new_obj.control_.lambda_.n_elem - 1)};
                    // try a finer grid with a larger "smallest lambda"
                    et_lambda_min =
                        std::pow(new_obj.control_.lambda_min_ratio_, 0.25) *
                        new_obj.output_.et_l1_lambda1_ /
                        std::max(new_obj.control_.ridge_alpha_,
                                 new_obj.control_.lambda_max_alpha_min_);
                    // in case it is actually smaller
                    if (min_lambda_i > et_lambda_min) {
                        // fall back to the default lambda generation
                        et_lambda_min = -1.0;
                    }
                }
            }
            l1_lambda0(i) = new_obj.output_.et_l1_lambda0_;
            l1_lambda1(i) = new_obj.output_.et_l1_lambda1_;
            // update active x
            const unsigned int p1_i{new_obj.p_data_->p1_ - p0};
            const unsigned int p0_i{new_obj.p_data_->p0_ - p0};
            const unsigned int et_nlambda{new_obj.output_.coef_.n_slices - 1};
            active_beta =
                new_obj.output_.coef_.slice(et_nlambda).head_rows(p1_i);
            arma::uvec pos_beta(p0_i);
            // get the indices of the selected predictors
            for (size_t j{0}; j < p0_i; ++j) {
                if (!active_beta.row(inter + j).is_zero()) {
                    pos_beta[j] = 1;
                }
            }
            active_idx0 = arma::find(pos_beta > 0);
            obj.output_.et_vs_ = obj.output_.et_vs_.elem(active_idx0);
            // verbose
            if (obj.control_.verbose_ > 0) {
                Rcpp::Rcout << "[ET] (stage " << i + 1
                            << ") Number of active predictors: "
                            << obj.output_.et_vs_.n_elem << "\n";
            }
            // record loss function
            obj.output_.loss_ = obj.output_.loss_(et_nlambda);
            obj.output_.penalty_ = obj.output_.penalty_(et_nlambda);
            obj.output_.objective_ = obj.output_.objective_(et_nlambda);
        }
        obj.output_.coef_ = arma::cube(obj.p_data_->p1_, obj.p_data_->k_ - 1, 1,
                                       arma::fill::zeros);
        if (obj.p_data_->intercept_) {
            obj.output_.coef_.slice(0).rows(obj.output_.et_vs_ + 1) =
                active_beta.rows(active_idx0 + 1);
            // intercept
            obj.output_.coef_.slice(0).row(0) = active_beta.row(0);
        } else {
            obj.output_.coef_.slice(0).rows(obj.output_.et_vs_) =
                active_beta.rows(active_idx0);
        }
        if (standardize0) {
            obj.force_rescale_coef();
        }
        obj.output_.et_l1_lambda0_vec_ = l1_lambda0;
        obj.output_.et_l1_lambda1_vec_ = l1_lambda1;
    }

    // estimate the prediction accuracy by cross-validation
    // for the model from the et-lasso procedure
    //! @param obj An Abclass object
    //! @param strata optional strata indicator variable for stratified
    //!     sampling in cross validation
    template <typename T> inline void et_cv_accuracy(T& obj)
    {
        // require data
        if (obj.empty_data()) {
            throw std::invalid_argument("Data required for et_cv_accuracy()");
        }
        // default to use y as the strata if stratified is true
        // and strata not specified
        if (obj.control_.cv_stratified_ &&
            obj.control_.cv_strata_.n_elem != obj.p_data_->n_obs_) {
            obj.control_.cv_strata_ = obj.p_data_->y_;
        }
        CrossValidation cv_obj{obj.p_data_->n_obs_, obj.control_.cv_nfolds_,
                               obj.control_.cv_strata_};
        obj.output_.cv_accuracy_ = arma::zeros(obj.control_.cv_nfolds_);
        // using DataType = std::remove_const_t<decltype(*obj.p_data_)>;
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
            // DataType new_data{obj.p_data_->k_, obj.p_data_->intercept_,
            // false};
            auto new_data{*obj.p_data_};
            new_data.set_data(std::move(train_x), std::move(train_y));
            new_data.set_obs_weights(std::move(train_weight));
            new_data.set_offset(std::move(train_offset));
            T new_obj{new_data, obj.control_};
            // alignment: 0 for alignment by fraction
            //            1 for alignment by lambda
            if (!obj.control_.custom_lambda_ &&
                obj.control_.cv_alignment_ == 0) {
                // reset lambda
                new_obj.control_.reg_lambda();
            }
            new_obj.control_.set_verbose(0);
            et_lambda(new_obj);
            obj.output_.cv_accuracy_(i) = new_obj.accuracy(
                new_obj.output_.coef_.slice(0), test_x, test_y, test_offset);
        }
        obj.output_.cv_accuracy_mean_ = arma::mean(obj.output_.cv_accuracy_);
        obj.output_.cv_accuracy_sd_ = arma::stddev(obj.output_.cv_accuracy_);
    }
} // namespace abclass

#endif
