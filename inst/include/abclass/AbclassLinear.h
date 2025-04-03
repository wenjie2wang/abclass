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

#ifndef ABCLASS_ABCLASS_LINEAR_H
#define ABCLASS_ABCLASS_LINEAR_H

#include <RcppArmadillo.h>
#include <stdexcept>

#include "Abclass.h"
#include "MarginLoss.h"

namespace abclass
{
    // angle-based classifiers with linear learning
    // T_x is intended to be arma::mat or arma::sp_mat
    // T_loss should be one of the loss function classes
    template <typename T_loss, typename T_x = arma::mat>
    class AbclassLinear : public Abclass<T_loss, T_x>
    {
    protected:
        // for the majorization-based algorithms
        double mm_lowerbound0_;
        arma::rowvec mm_lowerbound_;

        // cache
        double null_loss_;    // loss function for the null model
        double last_loss_;    // last loss
        double last_penalty_; // penalty of last coefficient estimates
        double last_obj_;     // last objective
        double last_eps_;     // last difference for checking congerence

        // given computed dloss_df
        inline arma::mat dloss_dbeta(const arma::mat& dloss_df_,
                                     const arma::vec& x_g) const
        {
            arma::mat dmat{dloss_df_};
            for (size_t j{0}; j < dmat.n_cols; ++j) {
                dmat.col(j) %= x_g;
            }
            return dmat;
        }

        inline arma::vec dloss_dbeta(const arma::vec& dloss_df_k,
                                     const arma::vec& x_g) const
        {
            arma::vec dvec{dloss_df_k};
            dvec %= x_g;
            return dvec;
        }

        // gradients for beta_g.
        inline arma::mat iter_dloss_dbeta(const unsigned int g)
        {
            return loss_fun_.dloss_dbeta(p_data_, iter_cache_, g);
        }

        // gradients for beta_gk
        inline arma::vec iter_dloss_dbeta(const unsigned int g,
                                          const unsigned int k)
        {
            return loss_fun_.dloss_dbeta(p_data_, iter_cache_, g, k);
        }

        // transfer coef for standardized data to coef for non-standardized data
        inline arma::mat rescale_coef(const arma::mat& beta) const
        {
            if (!p_data_->standardize_) {
                return beta;
            }
            arma::mat out{beta};
            if (p_data_->intercept_) {
                arma::rowvec tmp_row{p_data_->x_center_ / p_data_->x_scale_};
                // for each columns
                for (size_t k{0}; k < p_data_->km1_; ++k) {
                    arma::vec beta_k{beta.col(k)};
                    out(0, k) = beta(0, k) -
                        arma::as_scalar(tmp_row *
                                        beta_k.tail_rows(p_data_->p0_));
                    for (size_t l{1}; l < p_data_->p1_; ++l) {
                        out(l, k) = beta_k(l) / p_data_->x_scale_(l - 1);
                    }
                }
            } else {
                for (size_t k{0}; k < p_data_->km1_; ++k) {
                    for (size_t l{0}; l < p_data_->p0_; ++l) {
                        out(l, k) /= p_data_->x_scale_(l);
                    }
                }
            }
            return out;
        }

        // MM lowerbound used in coordinate-descent algorithm
        inline void set_mm_lowerbound()
        {
            double mm_lowerbound_factor{1.0};
            if constexpr (std::is_base_of_v<MarginLoss, T_loss>) {
                mm_lowerbound_factor = loss_fun_.mm_lowerbound();
            } else {
                mm_lowerbound_factor = loss_fun_.mm_lowerbound(p_data_->dk_);
            }
            if (p_data_->standardize_ && !p_data_->custom_obs_weight_) {
                mm_lowerbound_ = mm_lowerbound_factor *
                    arma::ones<arma::rowvec>(p_data_->p0_);
            } else if (p_data_->custom_obs_weight_) {
                T_x sqx{arma::square(p_data_->x_)};
                mm_lowerbound_ = mm_lowerbound_factor * p_data_->div_n_obs_ *
                    (p_data_->obs_weights_.t() * sqx);
            } else {
                T_x sqx{arma::square(p_data_->x_)};
                mm_lowerbound_ = mm_lowerbound_factor * arma::mean(sqx, 0);
            }
            if (p_data_->intercept_) {
                if (p_data_->custom_obs_weight_) {
                    mm_lowerbound0_ = mm_lowerbound_factor *
                        p_data_->div_n_obs_ * arma::accu(p_data_->obs_weights_);
                } else {
                    mm_lowerbound0_ = mm_lowerbound_factor;
                }
            }
        }

    public:
        AbclassLinear() {}

        // inherit constructors
        using Abclass<T_loss, T_x>::Abclass;

        // data members
        using Abclass<T_loss, T_x>::loss_fun_;
        using Abclass<T_loss, T_x>::iter_cache_;
        using Abclass<T_loss, T_x>::p_data_;
        using Abclass<T_loss, T_x>::p_control_;
        using Abclass<T_loss, T_x>::output_;
        using Abclass<T_loss, T_x>::control_;

        // function members
        using Abclass<T_loss, T_x>::empty_data;
        using Abclass<T_loss, T_x>::accuracy;
        using Abclass<T_loss, T_x>::predict_prob;
        using Abclass<T_loss, T_x>::predict_y;

        // prepare for model fitting
        inline void pre_fit() override
        {
            Abclass<T_loss, T_x>::pre_fit();
            set_coef_lower_limit(p_control_->lower_limit_);
            set_coef_upper_limit(p_control_->upper_limit_);
            scale_coef_limits();
        }

        // set coef lower limit
        inline void set_coef_lower_limit(const arma::mat& lower_limit)
        {
            if (lower_limit.n_elem == 0) {
                control_.lower_limit_ =
                    arma::mat(p_data_->p0_, p_data_->km1_,
                              arma::fill::value(-arma::datum::inf));
                return;
            }
            if (lower_limit.n_elem == 1) {
                double limit_value{arma::as_scalar(lower_limit)};
                if (limit_value > 0.0) {
                    throw std::invalid_argument(
                        "Lower limit cannot be positive!");
                }
                control_.lower_limit_ =
                    arma::mat(p_data_->p0_, p_data_->km1_,
                              arma::fill::value(limit_value));
                return;
            }
            if (lower_limit.n_rows != p_data_->p0_ ||
                lower_limit.n_cols != p_data_->km1_) {
                throw std::invalid_argument(
                    "Incorrect dimension of lower_limit!");
            }
            control_.lower_limit_ = lower_limit;
        }

        // set coef upper limit
        inline void set_coef_upper_limit(const arma::mat& upper_limit)
        {
            if (upper_limit.n_elem == 0) {
                control_.upper_limit_ =
                    arma::mat(p_data_->p0_, p_data_->km1_,
                              arma::fill::value(arma::datum::inf));
                return;
            }
            if (upper_limit.n_elem == 1) {
                double limit_value{arma::as_scalar(upper_limit)};
                if (limit_value < 0.0) {
                    throw std::invalid_argument(
                        "Upper limit cannot be negative!");
                }
                control_.upper_limit_ =
                    arma::mat(p_data_->p0_, p_data_->km1_,
                              arma::fill::value(limit_value));
                return;
            }
            if (upper_limit.n_rows != p_data_->p0_ ||
                upper_limit.n_cols != p_data_->km1_) {
                throw std::invalid_argument(
                    "Incorrect dimension of upper_limit!");
            }
            control_.upper_limit_ = upper_limit;
        }
        // scale coef limits
        inline void scale_coef_limits()
        {
            // scale only if needed
            if (!p_data_->standardize_) {
                return;
            }
            arma::uvec valid_x_idx{arma::find(p_data_->x_scale_ > 0)};
            for (size_t i : valid_x_idx) {
                control_.lower_limit_.row(i) *= p_data_->x_scale_(i);
                control_.upper_limit_.row(i) *= p_data_->x_scale_(i);
            }
        }

        // rescale the coefficients
        inline void force_rescale_coef()
        {
            // must know what you are doing
            for (size_t i{0}; i < output_.coef_.n_slices; ++i) {
                output_.coef_.slice(i) = rescale_coef(output_.coef_.slice(i));
            }
        }

        // linear predictor
        inline arma::mat
        linear_score(const arma::mat& beta, const T_x& x,
                     const arma::mat& offset = arma::mat()) const
        {
            arma::mat pred_mat;
            if (p_data_->intercept_) {
                pred_mat = x * beta.tail_rows(x.n_cols);
                pred_mat.each_row() += beta.row(0);
            } else {
                pred_mat = x * beta;
            }
            if (!offset.is_empty()) {
                // check the dimension of the offset term
                if (offset.n_rows != x.n_rows || offset.n_cols != beta.n_cols) {
                    throw std::invalid_argument(
                        "Inconsistent dimension of offset!");
                }
                pred_mat += offset;
            }
            return pred_mat;
        }

        // class conditional probability
        inline arma::mat
        predict_prob(const arma::mat& beta, const T_x& x,
                     const arma::mat& offset = arma::mat()) const
        {
            return predict_prob(linear_score(beta, x, offset));
        }

        // prediction based on the inner products
        inline arma::uvec predict_y(const arma::mat& beta, const T_x& x,
                                    const arma::mat& offset = arma::mat()) const
        {
            return predict_y(linear_score(beta, x, offset));
        }

        // accuracy for tuning
        inline double accuracy(const arma::mat& beta, const T_x& x,
                               const arma::uvec& y,
                               const arma::mat& offset = arma::mat()) const
        {
            return accuracy(linear_score(beta, x, offset), y);
        }
    };

} // namespace abclass

#endif /* ABCLASS_ABCLASS_LINEAR_H */
