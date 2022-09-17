//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2022 Eli Lilly and Company
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
#include "AbclassGroup.h"
#include "Control.h"
#include "utils.h"

namespace abclass
{
    template <typename T_loss, typename T_x>
    class AbclassGroupMCP : public AbclassGroup<T_loss, T_x>
    {
    protected:
        // data
        using AbclassGroup<T_loss, T_x>::dn_obs_;
        using AbclassGroup<T_loss, T_x>::km1_;
        using AbclassGroup<T_loss, T_x>::inter_;
        using AbclassGroup<T_loss, T_x>::mm_lowerbound_;
        using AbclassGroup<T_loss, T_x>::mm_lowerbound0_;

        // functions
        using AbclassGroup<T_loss, T_x>::loss_derivative;
        using AbclassGroup<T_loss, T_x>::gen_group_weight;
        using AbclassGroup<T_loss, T_x>::mm_gradient;
        using AbclassGroup<T_loss, T_x>::mm_gradient0;
        using AbclassGroup<T_loss, T_x>::gradient;
        using AbclassGroup<T_loss, T_x>::objective0;
        using AbclassGroup<T_loss, T_x>::set_mm_lowerbound;

        // common methods
        inline double mcp_group_penalty(const arma::rowvec& beta,
                                        const double lambda,
                                        const double gamma,
                                        const double ridge) const
        {
            double l2_beta { l2_norm(beta) };
            return mcp_penalty(l2_beta, lambda, gamma, ridge);
        }
        inline double mcp_penalty(const double theta,
                                  const double lambda,
                                  const double gamma,
                                  const double ridge) const
        {
            const double ridge_pen { 0.5 * ridge * theta * theta };
            if (theta < gamma * lambda) {
                return theta * (lambda - 0.5 * theta / gamma) + ridge_pen;
            }
            return 0.5 * gamma * lambda * lambda + ridge_pen;
        }

        inline double regularization(
            const arma::mat& beta,
            const double lambda,
            const double gamma,
            const double ridge,
            const arma::vec& group_weight
            ) const
        {
            double out { 0.0 };
            for (size_t g {0}; g < group_weight.n_elem; ++g) {
                out += mcp_group_penalty(
                    beta.row(g + inter_),
                    lambda * group_weight(g),
                    gamma,
                    ridge);
            }
            return out;
        }

        // objective function with regularization
        inline double objective(
            const arma::vec& inner,
            const arma::mat& beta,
            const double lambda,
            const double gamma,
            const double ridge,
            const arma::vec& group_weight
            ) const
        {
            return objective0(inner) +
                regularization(beta, lambda, gamma, ridge, group_weight);
        }

        // run one cycle of coordinate descent over a given active set
        inline void run_one_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::uvec& is_active,
                                         const double lambda,
                                         const double gamma,
                                         const double ridge,
                                         const bool update_active,
                                         const unsigned int verbose);

        // run complete cycles of GMD for a given active set and given lambda's
        inline void run_gmd_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::uvec& is_active,
                                         const double lambda,
                                         const double gamma,
                                         const double ridge,
                                         const bool varying_active_set,
                                         const unsigned int max_iter,
                                         const double epsilon,
                                         const unsigned int verbose);

    public:
        // inherit
        using AbclassGroup<T_loss, T_x>::AbclassGroup;
        using AbclassGroup<T_loss, T_x>::control_;
        using AbclassGroup<T_loss, T_x>::n_obs_;
        using AbclassGroup<T_loss, T_x>::p0_;
        using AbclassGroup<T_loss, T_x>::p1_;
        using AbclassGroup<T_loss, T_x>::ex_vertex_;
        using AbclassGroup<T_loss, T_x>::x_;
        using AbclassGroup<T_loss, T_x>::y_;
        using AbclassGroup<T_loss, T_x>::et_npermuted_;

        using AbclassGroup<T_loss, T_x>::lambda_max_;
        using AbclassGroup<T_loss, T_x>::custom_lambda_;
        using AbclassGroup<T_loss, T_x>::coef_;

        using AbclassGroup<T_loss, T_x>::rescale_coef;
        using AbclassGroup<T_loss, T_x>::set_group_weight;

        // for a sequence of lambda's
        inline void fit() override;

        inline void set_gamma(const double dgamma = 0.01)
        {
            if (mm_lowerbound_.empty()) {
                set_mm_lowerbound();
            }
            // exclude zeros lowerbounds from constant columns
            const double min_mg {
                mm_lowerbound_(arma::find(mm_lowerbound_ > 0.0)).min()
            };
            control_.gamma_ = dgamma + 1.0 / min_mg;
        }
    };

    // run one GMD cycle over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassGroupMCP<T_loss, T_x>::run_one_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::uvec& is_active,
        const double lambda,
        const double gamma,
        const double ridge,
        const bool update_active,
        const unsigned int verbose
        )
    {
        double ell_verbose { 0.0 }, obj_verbose { 0.0 }, reg_verbose { 0.0 };
        if (verbose > 2) {
            Rcpp::Rcout << "\nStarting values of beta:\n";
            Rcpp::Rcout << beta << "\n";
            Rcpp::Rcout << "The active set of beta:\n";
            Rcpp::Rcout << arma2rvec(is_active) << "\n";
        };
        if (verbose > 1) {
            obj_verbose = objective0(inner);
            reg_verbose = regularization(beta, lambda, gamma, ridge,
                                         control_.group_weight_);
            ell_verbose = obj_verbose + reg_verbose;
        }
        // for intercept
        if (control_.intercept_) {
            arma::rowvec delta_beta0 {
                - mm_gradient0(inner) / mm_lowerbound0_
            };
            beta.row(0) += delta_beta0;
            arma::vec tmp_du { ex_vertex_ * delta_beta0.t() };
            inner += tmp_du;
        }
        // for predictors
        for (size_t j {0}; j < p0_; ++j) {
            if (is_active(j) == 0) {
                continue;
            }
            const size_t j1 { j + inter_ };
            const double mj { mm_lowerbound_(j) }; // m_g
            // early exit for zero mj from constant columns
            if (isAlmostEqual(mj, 0.0)) {
                beta.row(j1).zeros();
                is_active(j) = 0;
                continue;
            }
            const arma::rowvec old_beta_j { beta.row(j1) };
            const arma::rowvec zj {
                - mm_gradient(inner, j) / mj + old_beta_j
            };
            const double lambda_j { lambda * control_.group_weight_(j) };
            const double zj2 { l2_norm(zj) };
            const double mj_ratio { 1.0 + ridge / mj }; // m_g' / m_g
            // update beta
            if (zj2 < gamma * lambda_j * mj_ratio) {
                double tmp { 1 - lambda_j / mj / zj2 };
                if (tmp > 0.0) {
                    double igamma_j { 1.0 / (gamma * mj) };
                    beta.row(j1) = tmp * zj / (mj_ratio - igamma_j);
                } else {
                    beta.row(j1).zeros();
                }
            } else {
                beta.row(j1) = zj / mj_ratio;
            }
            arma::rowvec delta_beta_j { (beta.row(j1) - old_beta_j) };
            arma::vec delta_vj { ex_vertex_ * delta_beta_j.t() };
            inner += x_.col(j) % delta_vj;
            if (update_active) {
                // check if it has been shrinkaged to zero
                if (l1_norm(beta.row(j1)) > 0.0) {
                    is_active(j) = 1;
                } else {
                    is_active(j) = 0;
                }
            }
        }
        if (verbose > 1) {
            double ell_old { ell_verbose };
            Rcpp::Rcout << "The objective function changed\n";
            Rprintf("  from %7.7f (obj. %7.7f + reg. %7.7f)\n",
                    ell_verbose, obj_verbose, reg_verbose);
            obj_verbose = objective0(inner);
            reg_verbose = regularization(beta, lambda, gamma, ridge,
                                         control_.group_weight_);
            ell_verbose = obj_verbose + reg_verbose;
            Rprintf("    to %7.7f (obj. %7.7f + reg. %7.7f)\n",
                    ell_verbose, obj_verbose, reg_verbose);
            if (ell_verbose > ell_old) {
                Rcpp::Rcout << "Warning: "
                            << "the objective function somehow increased\n";
            }
        }
    }

    // run CMD cycles over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassGroupMCP<T_loss, T_x>::run_gmd_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::uvec& is_active,
        const double lambda,
        const double gamma,
        const double ridge,
        const bool varying_active_set,
        const unsigned int max_iter,
        const double epsilon,
        const unsigned int verbose
        )
    {
        size_t i {0}, num_iter {0};
        // arma::mat beta0 { beta };
        double loss0 { objective0(inner) }, loss1 { loss0 };
        // use active-set if p > n ("helps when p >> n")
        if (varying_active_set) {
            arma::uvec is_active_strong { is_active },
                is_active_varying { is_active };
            if (verbose > 0) {
                Rcpp::Rcout << "The size of active set from strong rule: "
                            << l1_norm(is_active_strong)
                            << "\n";
            }
            while (i < max_iter) {
                // cycles over the active set
                size_t ii {0};
                while (ii < max_iter) {
                    num_iter = ii + 1;
                    Rcpp::checkUserInterrupt();
                    run_one_active_cycle(beta, inner, is_active_varying,
                                         lambda, gamma, ridge, true, verbose);
                    // if (rel_diff(beta0, beta) < epsilon) {
                    //     break;
                    // }
                    // beta0 = beta;
                    loss1 = objective0(inner);
                    if (std::abs(loss1 - loss0) < epsilon) {
                        break;
                    }
                    loss0 = loss1;
                    ++ii;
                }
                // run a full cycle over the converged beta
                run_one_active_cycle(beta, inner, is_active,
                                     lambda, gamma, ridge, true, verbose);
                ++num_iter;
                // check two active sets coincide
                if (l1_norm(is_active_varying - is_active) > 0) {
                    // if different, repeat this process
                    if (verbose > 0) {
                        Rcpp::Rcout << "Changed the active set from "
                                    << l1_norm(is_active_varying)
                                    << " to "
                                    << l1_norm(is_active)
                                    << " after "
                                    << num_iter
                                    << " iteration(s)\n";
                    }
                    is_active_varying = is_active;
                    // recover the active set
                    is_active = is_active_strong;
                    ++i;
                } else {
                    if (verbose > 0) {
                        Rcpp::Rcout << "Converged over the active set after "
                                    << num_iter
                                    << " iteration(s)\n";
                        Rcpp::Rcout << "The size of active set is "
                                    << l1_norm(is_active) << "\n";
                    }
                    break;
                }
                if (verbose > 0) {
                    msg("Outer loop reached the maximum number of iteratons.");
                }
            }
        } else {
            // regular coordinate descent
            while (i < max_iter) {
                Rcpp::checkUserInterrupt();
                ++num_iter;
                run_one_active_cycle(beta, inner, is_active,
                                     lambda, gamma, ridge, false, verbose);
                // if (rel_diff(beta0, beta) < epsilon) {
                //     break;
                // }
                // beta0 = beta;
                loss1 = objective0(inner);
                if (std::abs(loss1 - loss0) < epsilon) {
                    break;
                }
                loss0 = loss1;
                ++i;
            }
            if (verbose > 0) {
                if (num_iter < max_iter) {
                    Rcpp::Rcout << "Outer loop converged after "
                                << num_iter
                                << " iteration(s)\n";
                } else {
                    msg("Outer loop reached the maximum number of iteratons.");
                }
            }
        }
    }

    // for a sequence of lambda's
    // MCP_group_penalty(l2_norm(beta_j), group_weight_j * lambda, gamma)
    template <typename T_loss, typename T_x>
    inline void AbclassGroupMCP<T_loss, T_x>::fit()
    {
        // set the CMD lowerbound
        set_mm_lowerbound();
        // set group weight
        set_group_weight();
        // set gamma
        set_gamma(control_.dgamma_);
        arma::uvec penalty_group { arma::find(control_.group_weight_ > 0.0) };
        arma::uvec penalty_free { arma::find(control_.group_weight_ == 0.0) };
        // initialize
        arma::vec one_inner { arma::zeros(n_obs_) };
        arma::mat one_beta { arma::zeros(p1_, km1_) },
            one_grad_beta { one_beta };
        // need to determine lambda_max
        one_grad_beta = gradient(one_inner);
        // get large enough lambda for zero coefs in penalty_group
        lambda_max_ = 0.0;
        for (arma::uvec::iterator it { penalty_group.begin() };
             it != penalty_group.end(); ++it) {
            double tmp { l2_norm(one_grad_beta.row(*it)) };
            tmp /= control_.group_weight_(*it);
            tmp /= std::max(control_.alpha_, 1e-2);
            if (lambda_max_ < tmp) {
                lambda_max_ = tmp;
            }
        }
        // set up lambda sequence
        if (control_.lambda_.empty()) {
            double log_lambda_max { std::log(lambda_max_) };
            control_.lambda_ = arma::exp(
                arma::linspace(log_lambda_max,
                               log_lambda_max +
                               std::log(control_.lambda_min_ratio_),
                               control_.nlambda_)
                );
        } else {
            control_.lambda_ = arma::reverse(arma::unique(control_.lambda_));
            control_.nlambda_ = control_.lambda_.n_elem;
            custom_lambda_ = true;
        }
        // initialize the estimate cube
        coef_ = arma::cube(p1_, km1_, control_.lambda_.n_elem,
                           arma::fill::zeros);
        this->loss_wo_penalty_ = arma::zeros(control_.lambda_.n_elem);
        this->penalty_ = this->loss_wo_penalty_;

        double one_strong_rhs { 0.0 };
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::uvec is_active_strong { arma::zeros<arma::uvec>(p0_) };
        // only need to estimate beta not in the penalty group
        for (arma::uvec::iterator it { penalty_free.begin() };
             it != penalty_free.end(); ++it) {
            is_active_strong(*it) = 1;
        }
        double l1_lambda { control_.alpha_ * lambda_max_ };
        double l2_lambda { (1 - control_.alpha_) * lambda_max_ };
        run_gmd_active_cycle(one_beta,
                             one_inner,
                             is_active_strong,
                             l1_lambda,
                             control_.gamma_,
                             l2_lambda,
                             false,
                             control_.max_iter_,
                             control_.epsilon_,
                             control_.verbose_);
        // optim with varying active set when p > n
        double old_lambda { l1_lambda }; // for strong rule
        // main loop: for each lambda
        for (size_t li { 0 }; li < control_.lambda_.n_elem; ++li) {
            double lambda_li { control_.lambda_(li) };
            l1_lambda = control_.alpha_ * lambda_li;
            l2_lambda = (1 - control_.alpha_) * lambda_li;
            // early exit for lambda greater than lambda_max_
            // note that lambda is sorted
            if (lambda_li >= lambda_max_) {
                coef_.slice(li) = rescale_coef(one_beta);
                this->loss_wo_penalty_(li) = objective0(one_inner);
                this->penalty_(li) = regularization(
                    one_beta, l1_lambda, control_.gamma_,
                    l2_lambda, control_.group_weight_);
                continue;
            }
            // update active set by strong rule
            one_grad_beta = gradient(one_inner);
            for (arma::uvec::iterator it { penalty_group.begin() };
                 it != penalty_group.end(); ++it) {
                if (is_active_strong(*it) > 0) {
                    continue;
                }
                double one_strong_lhs { l2_norm(one_grad_beta.row(*it)) };
                one_strong_rhs = control_.group_weight_(*it) *
                    (control_.gamma_ / (control_.gamma_ - 1) *
                     (l1_lambda - old_lambda) + l1_lambda);
                if (one_strong_lhs >= one_strong_rhs) {
                    is_active_strong(*it) = 1;
                }
            }
            old_lambda = l1_lambda; // for next iteration
            bool kkt_failed { true };
            one_strong_rhs = l1_lambda;
            // eventually, strong rule will guess correctly
            while (kkt_failed) {
                arma::uvec is_active_strong_old { is_active_strong };
                arma::uvec is_strong_rule_failed {
                    arma::zeros<arma::uvec>(is_active_strong.n_elem)
                };
                // update beta
                run_gmd_active_cycle(one_beta,
                                     one_inner,
                                     is_active_strong,
                                     l1_lambda,
                                     control_.gamma_,
                                     l2_lambda,
                                     control_.varying_active_set_,
                                     control_.max_iter_,
                                     control_.epsilon_,
                                     control_.verbose_);
                if (control_.verbose_ > 0) {
                    msg("Checking the KKT condition for the null set.");
                }
                // check kkt condition
                for (arma::uvec::iterator it { penalty_group.begin() };
                     it != penalty_group.end(); ++it) {
                    if (is_active_strong_old(*it) > 0) {
                        continue;
                    }
                    if (l2_norm(mm_gradient(one_inner, *it)) >
                        one_strong_rhs * control_.group_weight_(*it)) {
                        // update active set
                        is_strong_rule_failed(*it) = 1;
                    }
                }
                if (arma::accu(is_strong_rule_failed) > 0) {
                    is_active_strong = is_active_strong_old ||
                        is_strong_rule_failed;
                    if (control_.verbose_ > 0) {
                        Rcpp::Rcout << "The strong rule failed for "
                                    << arma::accu(is_strong_rule_failed)
                                    << " group(s)\nThe size of old active set: "
                                    << l1_norm(is_active_strong_old)
                                    << "\nThe size of new active set: "
                                    << l1_norm(is_active_strong)
                                    << "\n";
                    }
                } else {
                    if (control_.verbose_ > 0) {
                        msg("The strong rule worked.\n");
                    }
                    kkt_failed = false;
                }
            }
            if (et_npermuted_ > 0) {
                if (control_.verbose_ > 0) {
                    msg("[ET] check if any pseudo-predictors was selected.");
                }
                // assume the last (permuted ) predictors are inactive
                arma::mat permuted_beta { one_beta.tail_rows(et_npermuted_) };
                if (! permuted_beta.is_zero(arma::datum::eps)) {
                    if (li == 0) {
                        msg("Warning: Fail to tune by ET-lasso; ",
                            "selected pseudo-predictor(s) by ",
                            "the largest lamabda.\n",
                            "Suggestion: increase 'lambda', ",
                            "'lambda_min_ratio' or 'nlambda'?");
                    } else {
                        coef_ = coef_.head_slices(li);
                    }
                    if (control_.verbose_ > 0) {
                        msg("[ET] selected pseudo-predictor(s).\n");
                    }
                    break;
                }
                if (control_.verbose_ > 0) {
                    msg("[ET] none of pseudo-predictors was selected.\n");
                }
                if (li == control_.lambda_.n_elem - 1) {
                    msg("Warning: Fail to tune by ET-lasso; ",
                        "no pseudo-predictors selected ",
                        "by the smallest lambda.\n",
                        "Suggestion: decrease 'lambda' or 'lambda_min_ratio'?");
                }
            }
            coef_.slice(li) = rescale_coef(one_beta);
            this->loss_wo_penalty_(li) = objective0(one_inner);
            this->penalty_(li) = regularization(
                one_beta, l1_lambda, control_.gamma_,
                l2_lambda, control_.group_weight_);
        }
    }

}  // abclass


#endif /* ABCLASS_ABCLASS_GROUP_MCP_H */
