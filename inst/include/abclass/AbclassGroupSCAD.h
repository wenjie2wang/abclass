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

#ifndef ABCLASS_ABCLASS_GROUP_SCAD_H
#define ABCLASS_ABCLASS_GROUP_SCAD_H

#include <RcppArmadillo.h>
#include "AbclassGroup.h"
#include "Control.h"
#include "utils.h"

namespace abclass
{
    template <typename T_loss, typename T_x>
    class AbclassGroupSCAD : public AbclassGroup<T_loss, T_x>
    {
    protected:
        // data
        using AbclassGroup<T_loss, T_x>::dn_obs_;
        using AbclassGroup<T_loss, T_x>::km1_;
        using AbclassGroup<T_loss, T_x>::inter_;
        using AbclassGroup<T_loss, T_x>::mm_lowerbound_;
        using AbclassGroup<T_loss, T_x>::mm_lowerbound0_;
        using AbclassGroup<T_loss, T_x>::null_loss_;

        // functions
        using AbclassGroup<T_loss, T_x>::loss_derivative;
        using AbclassGroup<T_loss, T_x>::mm_gradient;
        using AbclassGroup<T_loss, T_x>::mm_gradient0;
        using AbclassGroup<T_loss, T_x>::gradient;
        using AbclassGroup<T_loss, T_x>::objective0;
        using AbclassGroup<T_loss, T_x>::set_mm_lowerbound;

        // l1_lambda = alpha * lambda
        // l2_lambda = (1 - alpha) * lambda
        inline double penalty0(const double beta,
                               const double l1_lambda,
                               const double l2_lambda) const override
        {
            // assume beta >= 0.0
            const double ridge_pen { 0.5 * l2_lambda * beta * beta };
            if (beta > control_.ncv_gamma_ * l1_lambda) {
                return 0.5 * l1_lambda * l1_lambda * (control_.ncv_gamma_ + 1) +
                    ridge_pen;
            }
            if (beta > l1_lambda) {
                return (control_.ncv_gamma_ * l1_lambda * beta -
                        0.5 * (beta * beta + l1_lambda * l1_lambda)) /
                    (control_.ncv_gamma_ - 1) + ridge_pen;
            }
            return l1_lambda * beta + ridge_pen;
        }

        inline void set_gamma(const double kappa = 0.9) override
        {
            // kappa must be in (0, 1)
            if (is_le(kappa, 0.0) || is_ge(kappa, 1.0)) {
                throw std::range_error("The 'kappa' must be in (0, 1).");
            }
            if (mm_lowerbound_.empty()) {
                set_mm_lowerbound();
            }
            // exclude zeros lowerbounds from constant columns
            const double min_mg {
                mm_lowerbound_.elem(arma::find(mm_lowerbound_ > 0.0)).min()
            };
            control_.ncv_gamma_ = (1.0 + 1.0 / min_mg) / kappa;
        }

        inline double strong_rule_rhs(const double next_lambda,
                                      const double last_lambda) const override
        {
            return control_.ncv_gamma_ / (control_.ncv_gamma_ - 2.0) *
                (next_lambda - last_lambda) + next_lambda;
        }

        inline void update_beta_g(arma::mat::row_iterator beta_g_it,
                                  const arma::rowvec& u_g,
                                  const double l1_lambda_g,
                                  const double l2_lambda,
                                  const double m_g) override
        {
            arma::rowvec z_g { u_g / m_g };
            for (size_t i {0}; i < z_g.n_elem; ++i) {
                z_g[i] += *(std::next(beta_g_it, i));
            }
            const double z_g2 { l2_norm(z_g) };
            const double m_gp { m_g + l2_lambda }; // m_g'
            const double m_g_ratio { m_gp / m_g }; // m_g' / m_g > 1
            if (z_g2 > m_g_ratio * control_.ncv_gamma_ * l1_lambda_g) {
                for (size_t i {0}; i < z_g.n_elem; ++beta_g_it, ++i) {
                    *beta_g_it = z_g[i] / m_g_ratio;
                }
            } else if (z_g2 > (m_gp + 1.0) * l1_lambda_g / m_g) {
                const double numer { (control_.ncv_gamma_ - 1.0) * m_g };
                const double denom { (control_.ncv_gamma_ - 1.0) * m_gp - 1.0 };
                const double tmp { numer / denom *
                    (1.0 - control_.ncv_gamma_ * l1_lambda_g / numer / z_g2) };
                for (size_t i {0}; i < z_g.n_elem; ++beta_g_it, ++i) {
                    *beta_g_it = tmp * z_g[i];
                }
            } else {
                const double tmp {
                    (1.0 - l1_lambda_g / m_g / z_g2) / m_g_ratio
                };
                if (tmp > 0.0) {
                    for (size_t i {0}; i < z_g.n_elem; ++beta_g_it, ++i) {
                        *beta_g_it = tmp * z_g[i];
                    }
                } else {
                    for (size_t i {0}; i < z_g.n_elem; ++beta_g_it, ++i) {
                        *beta_g_it = 0.0;
                    }
                }
            }
        }

    public:
        // inherit
        using AbclassGroup<T_loss, T_x>::AbclassGroup;
        using AbclassGroup<T_loss, T_x>::control_;

    };

}  // abclass


#endif /* ABCLASS_ABCLASS_GROUP_SCAD_H */
