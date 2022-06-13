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

#ifndef ABCLASS_ABCLASS_GROUP_LASSO_H
#define ABCLASS_ABCLASS_GROUP_LASSO_H

#include <RcppArmadillo.h>
#include "AbclassGroup.h"
#include "Control.h"
#include "utils.h"

namespace abclass
{
    template <typename T_loss, typename T_x>
    class AbclassGroupLasso : public AbclassGroup<T_loss, T_x>
    {
    protected:
        // data
        using AbclassGroup<T_loss, T_x>::dn_obs_;
        using AbclassGroup<T_loss, T_x>::km1_;
        using AbclassGroup<T_loss, T_x>::p1_;
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
        inline double regularization(
            const arma::mat& beta,
            const double lambda,
            const arma::vec& group_weight
            ) const
        {
            double out { 0.0 };
            arma::uvec idx { arma::find(group_weight > 0.0) };
            arma::uvec::iterator it { idx.begin() };
            arma::uvec::iterator it_end { idx.end() };
            for (; it != it_end; ++it) {
                out += group_weight(*it) * l2_norm(beta.row(*it));
            }
            return lambda * out;
        }

        // objective function with regularization
        inline double objective(
            const arma::vec& inner,
            const arma::mat& beta,
            const double lambda,
            const arma::vec& group_weight
            ) const
        {
            return objective0(inner) +
                regularization(beta, lambda, group_weight);
        }

        // run one cycle of coordinate descent over a given active set
        inline void run_one_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::uvec& is_active,
                                         const double lambda,
                                         const bool update_active,
                                         const unsigned int verbose);

        // run complete cycles of GMD for a given active set and given lambda's
        inline void run_gmd_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::uvec& is_active,
                                         const double lambda,
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
        using AbclassGroup<T_loss, T_x>::vertex_;
        using AbclassGroup<T_loss, T_x>::x_;
        using AbclassGroup<T_loss, T_x>::y_;

        using AbclassGroup<T_loss, T_x>::lambda_max_;
        using AbclassGroup<T_loss, T_x>::custom_lambda_;
        using AbclassGroup<T_loss, T_x>::coef_;
        using AbclassGroup<T_loss, T_x>::num_iter_;

        using AbclassGroup<T_loss, T_x>::rescale_coef;
        using AbclassGroup<T_loss, T_x>::set_group_weight;

        // for a sequence of lambda's
        inline void fit() override;

    };

    // run one GMD cycle over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassGroupLasso<T_loss, T_x>::run_one_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::uvec& is_active,
        const double lambda,
        const bool update_active,
        const unsigned int verbose
        )
    {
        double ell_verbose { 0.0 }, obj_verbose { 0.0 }, reg_verbose { 0.0 };
        if (verbose > 2) {
            Rcpp::Rcout << "\nStarting values of beta:\n"
                        << beta << "\n"
                        << "\nThe active set of beta:\n"
                        << arma2rvec(is_active)
                        << "\n";
        };
        if (verbose > 1) {
            obj_verbose = objective0(inner);
            reg_verbose = regularization(beta, lambda, control_.group_weight_);
            ell_verbose = obj_verbose + reg_verbose;
        }
        // for intercept
        if (control_.intercept_) {
            arma::rowvec delta_beta0 {
                - mm_gradient0(inner) / mm_lowerbound0_
            };
            beta.row(0) += delta_beta0;
            arma::vec tmp_du { vertex_ * delta_beta0.t() };
            for (size_t i { 0 }; i < n_obs_; ++i) {
                inner[i] += tmp_du(y_[i]);
            }
        }
        // for predictors
        for (size_t j {0}; j < p0_; ++j) {
            if (is_active(j) == 0) {
                continue;
            }
            size_t j1 { j + inter_ };
            arma::rowvec old_beta_j { beta.row(j1) };
            double mj { mm_lowerbound_(j) };
            arma::rowvec uj {
                - mm_gradient(inner, j) + mj * beta.row(j1)
            };
            double lambda_j { lambda * control_.group_weight_(j) };
            double pos_part { 1 - lambda_j / l2_norm(uj) };
            // update beta
            if (pos_part <= 0.0) {
                beta.row(j1).zeros();
            } else {
                beta.row(j1) = uj * pos_part / mj;
            }
            for (size_t i {0}; i < n_obs_; ++i) {
                inner(i) += x_(i, j) *
                    arma::accu((beta.row(j1) - old_beta_j) %
                               vertex_.row(y_(i)));
            }
            if (update_active) {
                // check if it has been shrinkaged to zero
                if (arma::any(beta.row(j1) != 0.0)) {
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
            reg_verbose = regularization(beta, lambda, control_.group_weight_);
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
    inline void AbclassGroupLasso<T_loss, T_x>::run_gmd_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::uvec& is_active,
        const double lambda,
        const bool varying_active_set,
        const unsigned int max_iter,
        const double epsilon,
        const unsigned int verbose
        )
    {
        size_t i {0};
        arma::mat beta0 { beta };
        // use active-set if p > n ("helps when p >> n")
        if (varying_active_set) {
            arma::uvec is_active_strong { is_active },
                is_active_varying { is_active };
            if (verbose > 1) {
                Rcpp::Rcout << "The size of active set from strong rule: "
                            << l1_norm(is_active_strong)
                            << "\n";
            }
            while (i < max_iter) {
                // cycles over the active set
                size_t ii {0};
                while (ii < max_iter) {
                    run_one_active_cycle(beta, inner, is_active_varying,
                                         lambda, true, verbose);
                    if (rel_diff(beta0, beta) < epsilon) {
                        num_iter_ = ii + 1;
                        break;
                    }
                    beta0 = beta;
                    ii++;
                }
                // run a full cycle over the converged beta
                run_one_active_cycle(beta, inner, is_active,
                                     lambda, true, verbose);
                // check two active sets coincide
                if (l1_norm(is_active_varying - is_active) > 0) {
                    // if different, repeat this process
                    if (verbose > 1) {
                        Rcpp::Rcout << "Changed the active set from "
                                    << l1_norm(is_active_varying)
                                    << " to "
                                    << l1_norm(is_active)
                                    << " after "
                                    << num_iter_ + 1
                                    << " iteration(s)\n";
                    }
                    is_active_varying = is_active;
                    // recover the active set
                    is_active = is_active_strong;
                    i++;
                } else {
                    if (verbose > 1) {
                        Rcpp::Rcout << "Converged over the active set after "
                                    << num_iter_ + 1
                                    << " iteration(s)\n";
                        Rcpp::Rcout << "The size of active set is "
                                    << l1_norm(is_active) << "\n";
                    }
                    num_iter_ = i + 1;
                    break;
                }
            }
        } else {
            // regular coordinate descent
            while (i < max_iter) {
                run_one_active_cycle(beta, inner, is_active,
                                     lambda, false, verbose);
                if (rel_diff(beta0, beta) < epsilon) {
                    num_iter_ = i + 1;
                    break;
                }
                beta0 = beta;
                i++;
            }
        }
        if (verbose > 0) {
            if (num_iter_ < max_iter) {
                Rcpp::Rcout << "Converged after "
                            << num_iter_
                            << " iteration(s)\n";
            } else {
                msg("Reached the maximum number of iteratons.");
            }
        }
    }

    // for a sequence of lambda's
    // lambda * group_weight_j * l2_norm(beta_j)
    template <typename T_loss, typename T_x>
    inline void AbclassGroupLasso<T_loss, T_x>::fit()
    {
        // set the CMD lowerbound
        set_mm_lowerbound();
        // set group weight
        set_group_weight(control_.group_weight_);
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

        double one_strong_rhs { 0.0 };
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::uvec is_active_strong { arma::zeros<arma::uvec>(p0_) };
        // only need to estimate beta not in the penalty group
        for (arma::uvec::iterator it { penalty_free.begin() };
             it != penalty_free.end(); ++it) {
            is_active_strong(*it) = 1;
        }
        run_gmd_active_cycle(one_beta,
                             one_inner,
                             is_active_strong,
                             lambda_max_,
                             false,
                             control_.max_iter_,
                             control_.epsilon_,
                             control_.verbose_);
        // optim with varying active set when p > n
        double old_lambda { lambda_max_ }; // for strong rule
        // main loop: for each lambda
        for (size_t li { 0 }; li < control_.lambda_.n_elem; ++li) {
            double lambda_li { control_.lambda_(li) };
            // early exit for lambda greater than lambda_max_
            // note that lambda is sorted
            if (lambda_li >= lambda_max_) {
                coef_.slice(li) = rescale_coef(one_beta);
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
                    (2 * lambda_li - old_lambda);
                if (one_strong_lhs >= one_strong_rhs) {
                    is_active_strong(*it) = 1;
                }
            }
            old_lambda = lambda_li;
            bool kkt_failed { true };
            one_strong_rhs = lambda_li;
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
                                     lambda_li,
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
            coef_.slice(li) = rescale_coef(one_beta);
        }
    }

}


#endif /* ABCLASS_ABCLASS_GROUP_LASSO_H */
