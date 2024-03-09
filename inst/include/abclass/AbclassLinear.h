//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2024 Eli Lilly and Company
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
#include "Control.h"
#include "Simplex.h"

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
        double null_loss_;      // loss function for the null model

        // given computed dloss_df
        inline arma::mat dloss_dbeta(const arma::mat& dloss_df_,
                                     const arma::vec& x_g) const
        {
            arma::mat dmat { dloss_df_ };
            for (size_t j {0}; j < dmat.n_cols; ++j) {
                dmat.col(j) %= x_g;
            }
            return dmat;
        }

        inline arma::vec dloss_dbeta(const arma::vec& dloss_df_k,
                                     const arma::vec& x_g) const
        {
            arma::vec dvec { dloss_df_k };
            dvec %= x_g;
            return dvec;
        }

        // gradients for beta_g.
        inline arma::mat iter_dloss_dbeta(const unsigned int g)
        {
            return loss_fun_.dloss_dbeta(data_, control_.obs_weight_, g);
        }

        // gradients for beta_gk
        inline arma::vec iter_dloss_dbeta(const unsigned int g,
                                          const unsigned int k)
        {
            return loss_fun_.dloss_dbeta(data_, control_.obs_weight_, g, k);
        }

        // transfer coef for standardized data to coef for non-standardized data
        inline arma::mat rescale_coef(const arma::mat& beta) const
        {
            if (! control_.standardize_) {
                return beta;
            }
            arma::mat out { beta };
            if (control_.intercept_) {
                arma::rowvec tmp_row { data_.x_center_ / data_.x_scale_ };
                // for each columns
                for (size_t k { 0 }; k < data_.km1_; ++k) {
                    arma::vec beta_k { beta.col(k) };
                    out(0, k) = beta(0, k) -
                        arma::as_scalar(tmp_row * beta_k.tail_rows(data_.p0_));
                    for (size_t l { 1 }; l < data_.p1_; ++l) {
                        out(l, k) = beta_k(l) / data_.x_scale_(l - 1);
                    }
                }
            } else {
                for (size_t k { 0 }; k < data_.km1_; ++k) {
                    for (size_t l { 0 }; l < data_.p0_; ++l) {
                        out(l, k) /= data_.x_scale_(l);
                    }
                }
            }
            return out;
        }

        // MM lowerbound used in coordinate-descent algorithm
        inline void set_mm_lowerbound()
        {
            if (control_.intercept_) {
                mm_lowerbound0_ = loss_fun_.mm_lowerbound0(
                    data_, control_.obs_weight_);
            }
            mm_lowerbound_ = loss_fun_.mm_lowerbound(
                data_, control_.obs_weight_);
        }

    public:
        // inherits
        // constructors
        using Abclass<T_loss, T_x>::Abclass;

        // data members
        using Abclass<T_loss, T_x>::data_;
        using Abclass<T_loss, T_x>::control_;
        using Abclass<T_loss, T_x>::loss_fun_;

        // function members
        using Abclass<T_loss, T_x>::accuracy;
        using Abclass<T_loss, T_x>::predict_prob;
        using Abclass<T_loss, T_x>::predict_y;

        // estimates
        arma::cube coef_;       // p1_ x km1_ for linear learning in each slice

        // loss/penalty/objective functions along the solution path
        arma::vec loss_;
        arma::vec objective_;

        // rescale the coefficients
        inline void force_rescale_coef()
        {
            // must know what you are doing
            for (size_t i {0}; i < coef_.n_slices; ++i) {
                coef_.slice(i) = rescale_coef(coef_.slice(i));
            }
        }

        // linear predictor
        inline arma::mat linear_score(
            const arma::mat& beta,
            const T_x& x,
            const arma::mat& offset = arma::mat()
            ) const
        {
            arma::mat pred_mat;
            if (control_.intercept_) {
                pred_mat = x * beta.tail_rows(x.n_cols);
                pred_mat.each_row() += beta.row(0);
            } else {
                pred_mat = x * beta;
            }
            if (! offset.is_empty()) {
                // check the dimension of the offset term
                if (offset.n_rows != x.n_rows || offset.n_cols != beta.n_cols) {
                    throw std::range_error("Inconsistent dimension of offset!");
                }
                pred_mat += offset;
            }
            return pred_mat;
        }

        // class conditional probability
        inline arma::mat predict_prob(
            const arma::mat& beta,
            const T_x& x,
            const arma::mat& offset = arma::mat()
            ) const
        {
            return predict_prob(linear_score(beta, x, offset));
        }

        // prediction based on the inner products
        inline arma::uvec predict_y(
            const arma::mat& beta,
            const T_x& x,
            const arma::mat& offset = arma::mat()
            ) const
        {
            return predict_y(linear_score(beta, x, offset));
        }

        // accuracy for tuning
        inline double accuracy(
            const arma::mat& beta,
            const T_x& x,
            const arma::uvec& y,
            const arma::mat& offset = arma::mat()
            ) const
        {
            return accuracy(linear_score(beta, x, offset), y);
        }

    };

}  // abclass


#endif /* ABCLASS_ABCLASS_LINEAR_H */
