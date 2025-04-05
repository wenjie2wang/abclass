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

#ifndef ABCLASS_ABCLASS_H
#define ABCLASS_ABCLASS_H

#include <RcppArmadillo.h>

#include "Control.h"
#include "Data.h"

namespace abclass
{
    // base class for the angle-based large margin classifiers
    // T_x is intended to be arma::mat or arma::sp_mat
    // T_loss should be one of the loss function classes
    template <typename T_loss, typename T_x = arma::mat> class Abclass
    {
    protected:
        // loss function (with observational weights but no scaling of 1/n)
        inline double iter_loss() const
        {
            return loss_fun_.loss(p_data_, iter_cache_);
        }

        inline arma::mat iter_dloss_df() const
        {
            return loss_fun_.dloss_df(p_data_, iter_cache_);
        }

        inline arma::vec iter_dloss_df(const unsigned int k) const
        {
            return loss_fun_.dloss_df(p_data_, iter_cache_, k);
        }

        // prepare for model fitting
        inline void pre_fit()
        {
            if (empty_data()) {
                throw std::invalid_argument("Data cannot be empty.");
            }
            if (p_data_->x_.empty()) {
                throw std::invalid_argument("The 'x' cannot be empty.");
            }
            if (p_data_->y_.empty()) {
                throw std::invalid_argument("The 'y' cannot be empty.");
            }
            output_.data_ = p_data_;
            control_ = *p_control_;
        }

    public:
        T_loss loss_fun_;      // loss funciton class
        IterCache iter_cache_; // intermediate results
        // user-specified control parameters
        const Control* p_control_{nullptr};
        // training data container
        const Data<T_x>* p_data_{nullptr};
        // outputs
        Output<T_x> output_;
        Control control_;

        // constructors
        Abclass() {}

        explicit Abclass(const Control& control = Control())
            : p_control_(&control)
        {
        }

        Abclass(const Data<T_x>& data, const Control& control = Control())
        {
            set_data(data);
            set_control(control);
        }

        // setters
        inline void set_data(const Data<T_x>& data) { p_data_ = &data; }

        inline void set_control(const Control& control)
        {
            p_control_ = &control;
        }

        // check data
        inline bool empty_data() { return p_data_ == nullptr; }

        // conditional class probability
        inline arma::mat predict_prob(const arma::mat& pred_f) const
        {
            // pred_f: n x (k - 1) matrix
            // vertex_: (k - 1) x k matrix
            arma::mat out{pred_f * p_data_->vertex_}; // n x k
            out.each_col([&](arma::vec& a) { a = loss_fun_.prob_score_k(a); });
            arma::vec row_sums{arma::sum(out, 1)};
            out.each_col() /= row_sums;
            return out;
        }

        // predict categories for predicted classification functions
        inline arma::uvec predict_y(const arma::mat& pred_f) const
        {
            // pred_f: n x (k - 1) matrix
            // vertex_: (k - 1) x k matrix
            arma::mat out{pred_f * p_data_->vertex_}; // n x k
            return arma::index_max(out, 1);
        }

        // accuracy for tuning by cross-validation
        inline double accuracy(const arma::mat& pred_f,
                               const arma::uvec& y) const
        {
            // in case the decision functions are all zeros
            if (!p_data_->intercept_ && pred_f.is_zero()) {
                return 1.0 / static_cast<double>(p_data_->k_);
            }
            arma::uvec max_idx{predict_y(pred_f)};
            arma::uvec is_correct{max_idx == y};
            // note that y can be of length different than n_obs_
            return arma::sum(is_correct) / static_cast<double>(y.n_elem);
        }
    };

} // namespace abclass

#endif /* ABCLASS_ABCLASS_H */
