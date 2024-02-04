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

#ifndef ABCLASS_ABCLASS_MELLOWMAX_H
#define ABCLASS_ABCLASS_MELLOWMAX_H

#include <RcppArmadillo.h>
#include "AbclassCD.h"
#include "Mellowmax.h"
#include "Control.h"
#include "utils.h"

namespace abclass
{
    // the angle-based classifier with Mellowmax penalty
    // estimation by coordinate-majorization-descent algorithm
    template <typename T_loss, typename T_x>
    class AbclassMellowmax : public AbclassCD<T_loss, T_x>
    {
    protected:
        // data
        using AbclassCD<T_loss, T_x>::km1_;
        using AbclassCD<T_loss, T_x>::mm_lowerbound_;

        // functions
        using AbclassCD<T_loss, T_x>::gradient;
        using AbclassCD<T_loss, T_x>::mm_gradient;

        // Mellowmax with optional ridge penalty
        inline double penalty1(const arma::rowvec& beta,
                               const double l1_lambda,
                               const double l2_lambda) const override
        {
            const Mellowmax mlm { beta, control_.mellowmax_omega_ };
            double ridge_pen { 0.0 };
            // optional ridge penalty
            if (l2_lambda > 0) {
                ridge_pen = 0.5 * l2_lambda * l2_norm_square(beta);
            }
            return l1_lambda * mlm.value() + ridge_pen;
        }

        // determine the large-enough l1 lambda that results in zero coef's
        inline void set_lambda_max(const arma::vec& inner,
                                   const arma::uvec& positive_penalty) override
        {
            arma::mat one_grad_beta { arma::abs(gradient(inner)) };
            // get large enough lambda for zero coefs in positive_penalty
            l1_lambda_max_ = 0.0;
            lambda_max_ = 0.0;
            for (arma::uvec::const_iterator it { positive_penalty.begin() };
                 it != positive_penalty.end(); ++it) {
                double tmp { one_grad_beta.row(*it).max() };
                tmp /= control_.penalty_factor_(*it);
                tmp *= km1_;
                if (l1_lambda_max_ < tmp) {
                    l1_lambda_max_ = tmp;
                }
            }
            lambda_max_ =  l1_lambda_max_ /
                std::max(control_.ridge_alpha_, 1e-2);
        }

        // experimental
        // inline double strong_rule_rhs(const double next_lambda,
        //                               const double last_lambda) const override
        // {
        //     return 2.0 * next_lambda - last_lambda;
        // }

        inline void update_beta_gk(arma::mat& beta,
                                   arma::vec& inner,
                                   const size_t k,
                                   const size_t g,
                                   const size_t g1,
                                   const double l1_lambda,
                                   const double l2_lambda) override
        {
            const double old_beta_g1k { beta(g1, k) };
            const arma::vec v_k { get_vertex_y(k) };
            const arma::vec vk_xg { x_.col(g) % v_k };
            const double d_gk { mm_gradient(inner, vk_xg) };
            const double u_g { mm_lowerbound_(g) * old_beta_g1k - d_gk };
            const double l1_lambda_g0 {
                l1_lambda * control_.penalty_factor_(g)
            };
            // local approximation
            const Mellowmax mlm {
                beta.row(g1), control_.mellowmax_omega_
            };
            const arma::rowvec mlm_d { mlm.grad() };
            const double mlm_dk { mlm_d(k) };
            // const double mlm_d2k {
            //     control_.mellowmax_omega_ * mlm_dk * (1 - mlm_dk)
            // };
            const double l1_lambda_g {
                l1_lambda_g0 * (mlm_dk - mlm.mm_lb_ * std::abs(old_beta_g1k))
            };
            const double tmp { std::abs(u_g) - l1_lambda_g };
            if (tmp <= 0.0) {
                beta(g1, k) = 0.0;
            } else {
                const double numer { tmp * sign(u_g) };
                const double denom {
                    mm_lowerbound_(g) + l2_lambda +
                    l1_lambda_g0 * mlm.mm_lb_
                };
                // update beta
                beta(g1, k) = numer / denom;
            }
            // update inner
            inner += (beta(g1, k) - old_beta_g1k) * vk_xg;
        }

    public:
        // inherits constructors
        using AbclassCD<T_loss, T_x>::AbclassCD;

        // data members
        using AbclassCD<T_loss, T_x>::control_;
        using AbclassCD<T_loss, T_x>::x_;
        using AbclassCD<T_loss, T_x>::l1_lambda_max_;
        using AbclassCD<T_loss, T_x>::lambda_max_;

        // function members
        using AbclassCD<T_loss, T_x>::get_vertex_y;

    };

}  // abclass


#endif /* ABCLASS_ABCLASS_MELLOWMAX_H */
