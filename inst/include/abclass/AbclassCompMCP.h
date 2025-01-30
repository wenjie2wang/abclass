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

#ifndef ABCLASS_ABCLASS_COMPMCP_H
#define ABCLASS_ABCLASS_COMPMCP_H

#include <RcppArmadillo.h>
#include "AbclassCD.h"
#include "utils.h"

namespace abclass
{
    // the angle-based classifier with composite MCP penalty
    // estimation by coordinate-majorization-descent algorithm
    template <typename T_loss, typename T_x>
    class AbclassCompMCP : public AbclassCD<T_loss, T_x>
    {
    protected:
        // data
        using AbclassCD<T_loss, T_x>::mm_lowerbound_;
        using AbclassCD<T_loss, T_x>::last_eps_;

        // functions
        using AbclassCD<T_loss, T_x>::gradient;
        using AbclassCD<T_loss, T_x>::mm_gradient;
        using AbclassCD<T_loss, T_x>::set_mm_lowerbound;

        // inner penalty
        inline double penalty0(const double beta,
                               const double l1_lambda,
                               const double l2_lambda) const override
        {
            if (false) {        // non-sense
                return l2_lambda;
            }
            return mcp_penalty(beta, l1_lambda, control_.ncv_gamma_);
        }
        // outer penalty for each "group"
        inline double penalty1(const arma::rowvec& beta,
                               const double l1_lambda,
                               const double l2_lambda) const override
        {
            double out { 0.0 };
            double ridge_pen { 0.0 };
            for (size_t k {0}; k < beta.n_elem; ++k) {
                // inner penalty
                out += penalty0(std::abs(beta(k)), l1_lambda, l2_lambda);
                // optional ridge penalty
                ridge_pen += 0.5 * l2_lambda * beta(k) * beta(k);
            }
            // outer penalty
            out = mcp_penalty(
                out, l1_lambda,
                0.5 * data_.km1_ * control_.ncv_gamma_ * l1_lambda);
            return out + ridge_pen;
        }

        // determine the large-enough l1 lambda that results in zero coef's
        inline void set_lambda_max(const arma::uvec& positive_penalty) override
        {
            arma::mat one_grad_beta { arma::abs(gradient()) };
            // get large enough lambda for zero coefs in positive_penalty
            l1_lambda_max_ = 0.0;
            lambda_max_ = 0.0;
            for (arma::uvec::const_iterator it { positive_penalty.begin() };
                 it != positive_penalty.end(); ++it) {
                double tmp { one_grad_beta.row(*it).max() };
                tmp /= control_.penalty_factor_(*it);
                tmp = std::sqrt(tmp); // the difference
                if (l1_lambda_max_ < tmp) {
                    l1_lambda_max_ = tmp;
                }
            }
            lambda_max_ =  l1_lambda_max_ /
                std::max(control_.ridge_alpha_, control_.lambda_max_alpha_min_);
        }

        inline void set_gamma(const double kappa = 0.9) override
        {
            // kappa must be in (0, 1)
            if (is_le(kappa, 0.0) || is_ge(kappa, 1.0)) {
                throw std::range_error("The 'kappa' must be in (0, 1).");
            }
            control_.ncv_kappa_ = kappa;
            if (mm_lowerbound_.empty()) {
                set_mm_lowerbound();
            }
            // exclude zeros lowerbounds from constant columns
            const double min_mg {
                mm_lowerbound_.elem(arma::find(mm_lowerbound_ > 0.0)).min()
            };
            control_.ncv_gamma_ = 1.0 / min_mg / kappa;
        }

        // no strong rule so far
        // inline double strong_rule_rhs(const double next_lambda,
        //                               const double last_lambda) const override
        // {
        //     return (control_.ncv_gamma_ / (control_.ncv_gamma_ - 1) *
        //             (next_lambda - last_lambda) + next_lambda);
        // }

        inline void update_beta_gk(arma::mat& beta,
                                   const size_t k,
                                   const size_t g,
                                   const size_t g1,
                                   const double l1_lambda,
                                   const double l2_lambda) override
        {
            const double old_beta_g1k { beta(g1, k) };
            const double d_gk { mm_gradient(g, k) };
            // local approximation
            double inner_pen { 0.0 };
            for (size_t ki {0}; ki < data_.km1_; ++ki) {
                inner_pen += penalty0(std::abs(beta(g1, ki)),
                                      l1_lambda, 0.0);
            }
            const double local_factor {
                dmcp_penalty(inner_pen, l1_lambda,
                             0.5 * data_.km1_ * control_.ncv_gamma_ * l1_lambda) *
                dmcp_penalty(std::abs(beta(g1, k)), l1_lambda,
                             control_.ncv_gamma_)
            };
            const double l1_lambda_g {
                control_.penalty_factor_(g) * local_factor
            };
            const double m_g { mm_lowerbound_(g) };
            const double u_g { m_g * beta(g1, k) - d_gk };
            const double tmp { std::abs(u_g) - l1_lambda_g };
            if (tmp > 0.0) {
                const double numer { tmp * sign(u_g) };
                const double denom { m_g + l2_lambda };
                // update beta
                beta(g1, k) = std::max(
                    control_.lower_limit_(g, k),
                    std::min(control_.upper_limit_(g, k),
                             numer / denom));
            } else {
                beta(g1, k) = 0.0;
            }
            // update pred_f and inner
            const double delta_beta { beta(g1, k) - old_beta_g1k };
            if (delta_beta != 0.0) {
                if constexpr (std::is_base_of_v<MarginLoss, T_loss>) {
                    data_.iter_inner_ += delta_beta * data_.iter_vk_xg_;
                } else {
                    data_.iter_pred_f_.col(k) += delta_beta * data_.x_.col(g);
                }
                last_eps_ = std::max(last_eps_, m_g * delta_beta * delta_beta);
            }
        }

    public:
        // inherits constructors
        using AbclassCD<T_loss, T_x>::AbclassCD;

        // data members
        using AbclassCD<T_loss, T_x>::control_;
        using AbclassCD<T_loss, T_x>::data_;
        using AbclassCD<T_loss, T_x>::l1_lambda_max_;
        using AbclassCD<T_loss, T_x>::lambda_max_;

    };

}  // abclass


#endif /* ABCLASS_ABCLASS_COMPMCP_H */
