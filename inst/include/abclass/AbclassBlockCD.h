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

#ifndef ABCLASS_ABCLASS_BLOCKCD_H
#define ABCLASS_ABCLASS_BLOCKCD_H

#include <RcppArmadillo.h>

#include "AbclassCD.h"
#include "utils.h"

namespace abclass
{
    // angle-based classifier with coordinate-descent type of algorithms
    template <typename T_loss, typename T_x>
    class AbclassBlockCD : public AbclassCD<T_loss, T_x>
    {
    protected:
        // data members
        using AbclassCD<T_loss, T_x>::active_ncol_;
        using AbclassCD<T_loss, T_x>::mm_lowerbound0_;
        using AbclassCD<T_loss, T_x>::mm_lowerbound_;
        using AbclassCD<T_loss, T_x>::last_eps_;

        // function members
        using AbclassCD<T_loss, T_x>::dloss_dbeta;
        using AbclassCD<T_loss, T_x>::gradient;
        using AbclassCD<T_loss, T_x>::iter_dloss_dbeta;
        using AbclassCD<T_loss, T_x>::iter_dloss_df;
        using AbclassCD<T_loss, T_x>::iter_loss;
        using AbclassCD<T_loss, T_x>::mm_gradient0;
        using AbclassCD<T_loss, T_x>::mm_gradient;
        using AbclassCD<T_loss, T_x>::penalty0;
        using AbclassCD<T_loss, T_x>::strong_rule_lhs;
        using AbclassCD<T_loss, T_x>::strong_rule_rhs;

        // specifying that a blockwise CD should be used
        inline void set_active_ncol() override
        {
            active_ncol_ = 1;
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
        inline void set_lambda_max(const arma::uvec& positive_penalty) override
        {
            arma::mat one_grad_beta { gradient() };
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
                std::max(control_.ridge_alpha_, control_.lambda_max_alpha_min_);
        }

        inline void apply_strong_rule(
            arma::umat& is_active_strong,
            const double next_lambda,
            const double last_lambda,
            const arma::uvec positive_penalty
            ) override
        {
            // update active set by strong rule
            arma::mat one_grad_beta { gradient() };
            double one_strong_rhs { strong_rule_rhs(next_lambda, last_lambda) };
            for (arma::uvec::const_iterator it { positive_penalty.begin() };
                 it != positive_penalty.end(); ++it) {
                if (is_active_strong(*it, 0) > 0) {
                    continue;
                }
                double sr_lhs { strong_rule_lhs(one_grad_beta.row(*it)) };
                if (sr_lhs >= control_.penalty_factor_(*it) * one_strong_rhs) {
                    is_active_strong(*it, 0) = 1;
                }
            }
        }

        // kkt condition
        inline arma::umat is_kkt_failed(
            const arma::umat& is_active_strong,
            const arma::uvec& positive_penalty,
            const double l1_lambda) const override
        {
            arma::umat is_strong_rule_failed(arma::size(is_active_strong));
            arma::mat dloss_df_;
            if (positive_penalty.n_elem > 0) {
                dloss_df_ = iter_dloss_df();
            }
            for (arma::uvec::const_iterator it { positive_penalty.begin() };
                 it != positive_penalty.end(); ++it) {
                if (is_active_strong(*it, 0) > 0) {
                    continue;
                }
                const arma::vec x_g { data_.x_.col(*it) };
                const arma::mat dloss_dbeta_ { dloss_dbeta(dloss_df_, x_g) };
                const arma::rowvec tmp { arma::mean(dloss_dbeta_) };
                const double tmp_l2 { l2_norm(tmp) };
                if (tmp_l2 > l1_lambda * control_.penalty_factor_(*it)) {
                    // update active set
                    is_strong_rule_failed(*it, 0) = 1;
                }
            }
            return is_strong_rule_failed;
        }

        // group-wise update step for beta
        // default to group lasso
        inline virtual void update_beta_g(arma::mat& beta,
                                          const size_t g,
                                          const size_t g1,
                                          const double l1_lambda,
                                          const double l2_lambda)
        {
            const arma::rowvec old_beta_g1 { beta.row(g1) };
            const double mg { mm_lowerbound_(g) };
            const arma::rowvec ug { - mm_gradient(g) };
            const double l1_lambda_g {
                l1_lambda * control_.penalty_factor_(g)
            };
            const arma::rowvec z_mg { ug + mg * beta.row(g1) };
            const double pos_part { 1.0 - l1_lambda_g / l2_norm(z_mg) };
            if (pos_part > 0.0) {
                beta.row(g1) = arma::max(
                    control_.lower_limit_.row(g),
                    arma::min(control_.upper_limit_.row(g),
                              z_mg * (pos_part / (mg + l2_lambda))));
            } else {
                beta.row(g1).zeros();
            }
            // update pred_f and inner
            const arma::rowvec delta_beta { beta.row(g1) - old_beta_g1 };
            if (! delta_beta.is_zero()) {
                if constexpr (std::is_base_of_v<MarginLoss, T_loss>) {
                    data_.iter_inner_ += data_.iter_v_xg_ * delta_beta.t();
                } else {
                    data_.iter_pred_f_ += data_.x_.col(g) * delta_beta;
                }
                last_eps_ = std::max(last_eps_,
                                     arma::max(mg * delta_beta % delta_beta));
            }
        }

        inline void run_one_active_cycle(
            arma::mat& beta,
            arma::umat& is_active,
            const double l1_lambda,
            const double l2_lambda,
            const bool update_active,
            const unsigned int verbose
            ) override;

    public:
        // inherit constructors
        using AbclassCD<T_loss, T_x>::AbclassCD;

        // specified for template inheritance
        using AbclassCD<T_loss, T_x>::data_;
        using AbclassCD<T_loss, T_x>::control_;
        using AbclassCD<T_loss, T_x>::loss_fun_;
        using AbclassCD<T_loss, T_x>::l1_lambda_max_;
        using AbclassCD<T_loss, T_x>::lambda_max_;
        using AbclassCD<T_loss, T_x>::n_iter_;

    };

    // run one group-wise update step over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassBlockCD<T_loss, T_x>::run_one_active_cycle(
        arma::mat& beta,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
        const bool update_active,
        const unsigned int verbose
        )
    {
        last_eps_ = 0.0;
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
                - mm_gradient0() / mm_lowerbound0_
            };
            if (! delta_beta0.is_zero()) {
                beta.row(0) += delta_beta0;
                // update pred_f_ and inner_
                if constexpr (std::is_base_of_v<MarginLoss, T_loss>) {
                    data_.iter_inner_ += data_.ex_vertex_ * delta_beta0.t();
                } else {
                    data_.iter_pred_f_.each_row() += delta_beta0;
                }
                last_eps_ = std::max(
                    last_eps_,
                    arma::max(mm_lowerbound0_ * (delta_beta0 % delta_beta0)));
            }
        }
        // for predictors
        for (size_t g {0}; g < data_.p0_; ++g) {
            const size_t g1 { g + data_.inter_ };
            if (is_active(g) == 0) {
                continue;
            }
            // update beta and inner
            update_beta_g(beta, g, g1, l1_lambda, l2_lambda);
            // update active
            if (update_active) {
                // check if it has been shrinkaged to zero
                if (beta.row(g1).is_zero()) {
                    is_active(g) = 0;
                }
                // is_active(g) must be one to get here
            }
        }
        ++n_iter_;
    }



}  // abclass


#endif /* ABCLASS_ABCLASS_BLOCKCD_H */
