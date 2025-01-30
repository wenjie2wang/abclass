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

#ifndef ABCLASS_ABCLASS_GROUP_MCP_H
#define ABCLASS_ABCLASS_GROUP_MCP_H

#include <RcppArmadillo.h>
#include "AbclassBlockCD.h"
#include "utils.h"

namespace abclass
{
    template <typename T_loss, typename T_x>
    class AbclassGroupMCP : public AbclassBlockCD<T_loss, T_x>
    {
    protected:
        // data
        using AbclassBlockCD<T_loss, T_x>::mm_lowerbound_;
        using AbclassBlockCD<T_loss, T_x>::last_eps_;

        // functions
        using AbclassBlockCD<T_loss, T_x>::mm_gradient;
        using AbclassBlockCD<T_loss, T_x>::set_mm_lowerbound;

        inline double penalty0(const double beta,
                               const double l1_lambda,
                               const double l2_lambda) const override
        {
            const double ridge_pen { 0.5 * l2_lambda * beta * beta };
            if (beta < control_.ncv_gamma_ * l1_lambda) {
                return beta * (l1_lambda - 0.5 * beta / control_.ncv_gamma_) +
                    ridge_pen;
            }
            return 0.5 * control_.ncv_gamma_ * l1_lambda * l1_lambda +
                ridge_pen;
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

        inline double strong_rule_rhs(const double next_lambda,
                                      const double last_lambda) const override
        {
            return (control_.ncv_gamma_ / (control_.ncv_gamma_ - 1) *
                    (next_lambda - last_lambda) + next_lambda);
        }

        inline void update_beta_g(arma::mat& beta,
                                  const size_t g,
                                  const size_t g1,
                                  const double l1_lambda,
                                  const double l2_lambda) override
        {
            const arma::rowvec old_beta_g1 { beta.row(g1) };
            const double m_g { mm_lowerbound_(g) };
            const arma::rowvec u_g { - mm_gradient(g) };
            const double l1_lambda_g {
                l1_lambda * control_.penalty_factor_(g)
            };
            const arma::rowvec z_g { u_g / m_g + beta.row(g1)};
            const double z_g2 { l2_norm(z_g) };
            const double m_gp { m_g + l2_lambda }; // m_g'
            const double m_g_ratio { m_gp / m_g }; // m_g' / m_g > 1
            if (z_g2 < control_.ncv_gamma_ * l1_lambda_g * m_g_ratio) {
                const double tmp { 1.0 - l1_lambda_g / m_g / z_g2 };
                if (tmp > 0.0) {
                    const double igamma_g { 1.0 / control_.ncv_gamma_ / m_g };
                    const double rhs { tmp / (m_g_ratio - igamma_g) };
                    beta.row(g1) = arma::max(
                        control_.lower_limit_.row(g),
                        arma::min(control_.upper_limit_.row(g),
                                  rhs * z_g));
                } else {
                    beta.row(g1).zeros();
                }
            } else {
                beta.row(g1) = arma::max(
                    control_.lower_limit_.row(g),
                    arma::min(control_.upper_limit_.row(g),
                              z_g / m_g_ratio));
            }
            // update pred_f and inner
            const arma::rowvec delta_beta { beta.row(g1) - old_beta_g1 };
            if (! delta_beta.is_zero()) {
                if constexpr (std::is_base_of_v<MarginLoss, T_loss>) {
                    data_.iter_inner_ += data_.iter_v_xg_ * delta_beta.t();
                } else {
                    data_.iter_pred_f_ += data_.x_.col(g) * delta_beta;
                }
                last_eps_ = std::max(
                    last_eps_, arma::max(m_g * (delta_beta % delta_beta)));
            }
        }

    public:
        // inherit
        using AbclassBlockCD<T_loss, T_x>::AbclassBlockCD;

        // data members
        using AbclassBlockCD<T_loss, T_x>::control_;
        using AbclassBlockCD<T_loss, T_x>::data_;

    };

}  // abclass


#endif /* ABCLASS_ABCLASS_GROUP_MCP_H */
