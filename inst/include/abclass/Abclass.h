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

#include <stdexcept>

#include "Control.h"
#include "Simplex.h"
#include "utils.h"

namespace abclass
{
    // base class for the angle-based large margin classifiers
    // T_x is intended to be arma::mat or arma::sp_mat
    // T_loss should be one of the loss function classes
    template <typename T_loss, typename T_x = arma::mat>
    class Abclass
    {
    protected:

        // loss function (with observational weights but no scaling of 1/n)
        inline double iter_loss() const
        {
            return loss_fun_.loss(data_, control_.obs_weight_);
        }

        inline arma::mat iter_dloss_df() const
        {
            return loss_fun_.dloss_df(data_, control_.obs_weight_);
        }

        inline arma::vec iter_dloss_df(const unsigned int k) const
        {
            return loss_fun_.dloss_df(data_, control_.obs_weight_, k);
        }

    public:
        Simplex2<T_x> data_;    // data container
        Control control_;       // control parameters
        T_loss loss_fun_;       // loss funciton class

        // default constructor
        Abclass() {}

        // for using prediction functions
        explicit Abclass(const unsigned k)
        {
            set_k(k);
        }

        // main constructor
        Abclass(const T_x& x,
                const arma::uvec& y,
                const Control& control = Control()) :
            control_ { control }
        {
            set_data(x, y);
            set_weight(control_.obs_weight_);
            set_offset(control_.offset_);
        }

        // k should be set from y
        inline void set_k(const unsigned int k)
        {
            data_ = Simplex2<T_x>(k);
        }

        // enforce k instead of setting it by y
        inline void enforce_k(const unsigned int k)
        {
            // assume y_ is set, update vertex's
            data_.update_k(k);
            set_y(data_.y_);
        }

        // setter
        inline void set_data(const T_x& x,
                             const arma::uvec& y)
        {
            // assume y in {0, ..., k-1}
            // assume binary classification if y only takes zero
            set_k(std::max(2U, arma::max(y + 1)));
            set_y(y);
            set_x(x);
        }

        inline void set_y(const arma::uvec& y)
        {
            data_.n_obs_ = y.n_elem;
            data_.div_n_obs_ = 1.0 / static_cast<double>(data_.n_obs_);
            data_.y_ = y;
            // assume k is set
            if (control_.owl_reward_.is_empty()) {
                data_.set_ex_vertex(y);
                return;
            }
            // for outcome-weighted learning
            if (control_.owl_reward_.n_elem != data_.n_obs_) {
                throw std::range_error("Inconsistent length of reward!");
            }
            data_.set_ex_vertex(y, sign(control_.owl_reward_));
        }

        inline void set_x(const T_x& x)
        {
            // assume y has been set correspondingly or not initialized
            if (! data_.y_.is_empty() && x.n_rows != data_.y_.n_elem) {
                throw std::range_error(
                    "The number of observations in X and y differs!");
            }
            data_.x_ = x;
            data_.inter_ = static_cast<unsigned int>(control_.intercept_);
            data_.p0_ = data_.x_.n_cols;
            data_.p1_ = data_.p0_ + data_.inter_;
            if (control_.standardize_) {
                if (control_.intercept_) {
                    data_.x_center_ = arma::mean(data_.x_);
                } else {
                    data_.x_center_ = arma::zeros<arma::rowvec>(data_.p0_);
                }
                data_.x_scale_ = col_sd(data_.x_);
                for (size_t j {0}; j < data_.p0_; ++j) {
                    if (data_.x_scale_(j) > 0) {
                        data_.x_.col(j) =
                            (data_.x_.col(j) - data_.x_center_(j)) /
                            data_.x_scale_(j);
                    } else {
                        data_.x_.col(j).zeros();
                        // make scale(j) nonzero for rescaling
                        data_.x_scale_(j) = - 1.0;
                    }
                }
            } else {
                data_.x_scale_ = col_sd(data_.x_);
            }
            data_.x_skip_ = arma::find(data_.x_scale_ <= 0.0);
        }

        inline void set_intercept(const bool intercept)
        {
            control_.intercept_ = intercept;
        }

        inline void set_standardize(const bool standardize)
        {
            control_.standardize_ = standardize;
        }

        inline void set_weight(const arma::vec& weight)
        {
            if (weight.n_elem != data_.n_obs_) {
                control_.obs_weight_ = arma::ones(data_.n_obs_);
                control_.custom_obs_weight_ = false;
            } else {
                control_.obs_weight_ = weight /
                    (arma::accu(weight) * data_.div_n_obs_);
                control_.custom_obs_weight_ = true;
            }
        }

        inline void set_offset(const arma::mat& offset)
        {
            if (offset.n_elem == 0 || offset.is_zero()) {
                control_.offset_ = arma::mat();
                control_.has_offset_ = false;
                return;
            }
            if (offset.n_rows != data_.n_obs_ || offset.n_cols != data_.km1_) {
                throw std::range_error("Inconsistent length of offsets!");
            }
            control_.offset_ = offset;
            control_.has_offset_ = true;
        }

        // conditional class probability
        inline virtual arma::mat predict_prob(const arma::mat& pred_f) const
        {
            // pred_f: n x (k - 1) matrix
            // vertex_: (k - 1) x k matrix
            arma::mat out { pred_f * data_.vertex_ }; // n x k
            out.each_col([&](arma::vec& a) {
                a = loss_fun_.prob_score_k(a);
            });
            arma::vec row_sums { arma::sum(out, 1) };
            out.each_col() /= row_sums;
            return out;
        }

        // predict categories for predicted classification functions
        inline arma::uvec predict_y(const arma::mat& pred_f) const
        {
            // pred_f: n x (k - 1) matrix
            // vertex_: (k - 1) x k matrix
            arma::mat out { pred_f * data_.vertex_ }; // n x k
            return arma::index_max(out, 1);
        }

        // accuracy for tuning by cross-validation
        inline double accuracy(const arma::mat& pred_f,
                               const arma::uvec& y) const
        {
            // in case the decision functions are all zeros
            if (! control_.intercept_ && pred_f.is_zero()) {
                return 1.0 / static_cast<double>(data_.k_);
            }
            arma::uvec max_idx { predict_y(pred_f) };
            arma::uvec is_correct { max_idx == y };
            // note that y can be of length different than n_obs_
            return arma::sum(is_correct) / static_cast<double>(y.n_elem);
        }

    };

}

#endif /* ABCLASS_ABCLASS_H */
