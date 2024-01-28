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

#ifndef ABCLASS_ABCLASS_BLOCKCD_H
#define ABCLASS_ABCLASS_BLOCKCD_H

#include <utility>

#include <RcppArmadillo.h>

#include "AbclassCD.h"
#include "Control.h"
#include "utils.h"

namespace abclass
{
    // angle-based classifier with coordinate-descent type of algorithms
    template <typename T_loss, typename T_x>
    class AbclassBlockCD : public AbclassCD<T_loss, T_x>
    {
    protected:

        // data members
        using AbclassCD<T_loss, T_x>::dn_obs_;
        using AbclassCD<T_loss, T_x>::inter_;
        using AbclassCD<T_loss, T_x>::km1_;
        using AbclassCD<T_loss, T_x>::mm_lowerbound0_;
        using AbclassCD<T_loss, T_x>::mm_lowerbound_;
        using AbclassCD<T_loss, T_x>::l1_lambda_max_;
        using AbclassCD<T_loss, T_x>::lambda_max_;

        // function members
        using AbclassCD<T_loss, T_x>::get_vertex_y;
        using AbclassCD<T_loss, T_x>::gradient;
        using AbclassCD<T_loss, T_x>::mm_gradient0;
        using AbclassCD<T_loss, T_x>::mm_gradient;
        using AbclassCD<T_loss, T_x>::penalty0;

        // specifying that a blockwise CD should be used
        inline size_t get_active_ncol() const override
        {
            return 1;
        }

        // group penalty for coefficients of one covariate
        inline double penalty1(const arma::rowvec& beta,
                               const double l1_lambda,
                               const double l2_lambda) const override
        {
            const double l2_beta { l2_norm(beta) };
            return penalty0(l2_beta, l1_lambda, l2_lambda);
        }

        // determine the large-enough l1 lambda that results in zero coef's
        inline void set_lambda_max(const arma::vec& inner,
                                   const arma::uvec& positive_penalty) override
        {
            arma::mat one_grad_beta { gradient(inner) };
            // get large enough lambda for zero coefs in penalty_group
            l1_lambda_max_ = 0.0;
            lambda_max_ = 0.0;
            for (arma::uvec::const_iterator it { positive_penalty.begin() };
                 it != positive_penalty.end(); ++it) {
                double tmp { l2_norm(one_grad_beta.row(*it)) };
                tmp /= control_.penalty_factor_(*it);
                if (l1_lambda_max_ < tmp) {
                    l1_lambda_max_ = tmp;
                }
            }
            lambda_max_ =  l1_lambda_max_ /
                std::max(control_.ridge_alpha_, 1e-2);
        }

        // group-wise update step for beta
        // default to group lasso
        inline virtual void update_beta_g(arma::mat& beta,
                                          arma::vec& inner,
                                          const size_t g,
                                          const size_t g1,
                                          const double l1_lambda,
                                          const double l2_lambda)
        {
            const arma::rowvec old_beta_g1 { beta.row(g1) };
            const double mg { mm_lowerbound_(g) };
            const arma::rowvec ug { - mm_gradient(inner, g) };
            const double l1_lambda_g {
                l1_lambda * control_.penalty_factor_(g)
            };
            const arma::rowvec z_mg { ug + mg * beta.row(g1) };
            const double pos_part { 1.0 - l1_lambda_g / l2_norm(z_mg) };
            if (pos_part > 0.0) {
                beta.row(g1) = z_mg * (pos_part / (mg + l2_lambda));
            } else {
                beta.row(g1).zeros();
            }
            // update inner
            const arma::rowvec delta_beta_j { beta.row(g1) - old_beta_g1 };
            const arma::vec delta_vj { ex_vertex_ * delta_beta_j.t() };
            inner += x_.col(g) % delta_vj;
        }

        inline void run_one_full_cycle(
            arma::mat& beta,
            arma::vec& inner,
            const double l1_lambda,
            const double l2_lambda,
            const unsigned int verbose
            ) override;

        inline void run_one_active_cycle(
            arma::mat& beta,
            arma::vec& inner,
            arma::umat& is_active,
            const double l1_lambda,
            const double l2_lambda,
            const bool update_active,
            const unsigned int verbose
            ) override;

    public:

        // inherit constructors
        using AbclassCD<T_loss, T_x>::AbclassCD;

        // specifics for template inheritance
        // from Abclass
        using AbclassCD<T_loss, T_x>::control_;
        using AbclassCD<T_loss, T_x>::ex_vertex_;
        using AbclassCD<T_loss, T_x>::loss_;
        using AbclassCD<T_loss, T_x>::n_obs_;
        using AbclassCD<T_loss, T_x>::objective_;
        using AbclassCD<T_loss, T_x>::p0_;
        using AbclassCD<T_loss, T_x>::p1_;
        using AbclassCD<T_loss, T_x>::penalty_;
        using AbclassCD<T_loss, T_x>::x_;

        // from AbclassLinear
        using AbclassCD<T_loss, T_x>::coef_;
        using AbclassCD<T_loss, T_x>::set_mm_lowerbound;

    };

    // one full cycle for coordinate-descent
    template <typename T_loss, typename T_x>
    inline void AbclassBlockCD<T_loss, T_x>::run_one_full_cycle(
        arma::mat& beta,
        arma::vec& inner,
        const double l1_lambda,
        const double l2_lambda,
        const unsigned int verbose
        )
    {
        if (verbose > 2) {
            Rcpp::Rcout << "\nStarting values of beta:\n";
            Rcpp::Rcout << beta << "\n";
        };
        // for intercept
        if (control_.intercept_) {
            arma::rowvec delta_beta0 {
                - mm_gradient0(inner) / mm_lowerbound0_
            };
            beta.row(0) += delta_beta0;
            arma::vec tmp_du { ex_vertex_ * delta_beta0.t() };
            inner += tmp_du;
        }
        // predictors
        for (size_t g { 0 }; g < p0_; ++g) {
            const size_t g1 { g + inter_ };
            // update beta and inner
            update_beta_g(beta, inner, g, g1, l1_lambda, l2_lambda);
        }
    }

    // run one group-wise update step over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassBlockCD<T_loss, T_x>::run_one_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
        const bool update_active,
        const unsigned int verbose
        )
    {
        if (verbose > 2) {
            Rcpp::Rcout << "\nStarting values of beta:\n"
                        << beta << "\n"
                        << "\nThe active set of beta:\n"
                        << arma2rvec(is_active)
                        << "\n";
        };
        // for intercept
        if (control_.intercept_) {
            const arma::rowvec delta_beta0 {
                - mm_gradient0(inner) / mm_lowerbound0_
            };
            beta.row(0) += delta_beta0;
            const arma::vec tmp_du { ex_vertex_ * delta_beta0.t() };
            inner += tmp_du;
        }
        // for predictors
        for (size_t g {0}; g < p0_; ++g) {
            if (is_active(g) == 0) {
                continue;
            }
            const size_t g1 { g + inter_ };
            // update beta and inner
            update_beta_g(beta, inner, g, g1, l1_lambda, l2_lambda);
            // update active
            if (update_active) {
                // check if it has been shrinkaged to zero
                if (l1_norm(beta.row(g1)) > 0.0) {
                    is_active(g) = 1;
                } else {
                    is_active(g) = 0;
                }
            }
        }
    }



}  // abclass


#endif /* ABCLASS_ABCLASS_BLOCKCD_H */
