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
#include "Abclass.h"
#include "utils.h"

namespace abclass
{
    class AbclassGroupLasso : public Abclass
    {
    protected:
        // for groupwise majorization descent
        arma::rowvec gmd_lowerbound_; // 1 by p1_

        // pure virtual functions
        virtual void set_gmd_lowerbound() = 0;
        virtual double objective0(const arma::vec& inner) const = 0;

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

        // define gradient function for j-th predictor
        inline arma::rowvec gmd_gradient(const arma::vec& inner,
                                         const unsigned int j) const
        {
            arma::vec inner_grad { loss_derivative(inner) };
            arma::rowvec out { arma::zeros<arma::rowvec>(km1_) };
            for (size_t i {0}; i < n_obs_; ++i) {
                out += obs_weight_[i] * inner_grad[i] * x_(i, j) *
                    vertex_.row(y_[i]);
            }
            return out / dn_obs_;
        }

        // gradient matrix for beta
        inline arma::mat gradient(const arma::vec& inner) const
        {
            arma::mat out { arma::zeros(p1_, km1_) };
            arma::vec inner_grad { loss_derivative(inner) };
            for (size_t j {0}; j < p1_; ++j) {
                arma::rowvec tmp { arma::zeros<arma::rowvec>(km1_) };
                for (size_t i {0}; i < n_obs_; ++i) {
                    tmp += obs_weight_[i] * inner_grad[i] * x_(i, j) *
                        vertex_.row(y_[i]);
                }
                out.row(j) = tmp;
            }
            return out / dn_obs_;
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
                                         const double rel_tol,
                                         const unsigned int verbose);

    public:

        // inherit constructors
        using Abclass::Abclass;

        // regularization
        // the "big" enough lambda => zero coef unless alpha = 0
        double lambda_max_;
        arma::vec lambda_;        // lambda sequence
        arma::vec group_weight_;  // adaptive weights for each group

        // estimates
        arma::cube coef_;         // p1_ by km1_

        // tuning by cross-validation
        arma::mat cv_accuracy_;
        arma::vec cv_accuracy_mean_;
        arma::vec cv_accuracy_sd_;

        // control
        double rel_tol_;          // relative tolerance for convergence check
        unsigned int max_iter_;   // maximum number of iterations
        bool varying_active_set_; // if active set should be adaptive

        // cache
        unsigned int num_iter_;   // number of CMD cycles till convergence

        // for a sequence of lambda's
        inline void fit(const arma::vec& lambda,
                        const unsigned int nlambda,
                        const double lambda_min_ratio,
                        const arma::vec& group_weight,
                        const unsigned int max_iter,
                        const double rel_tol,
                        const bool varying_active_set,
                        const unsigned int verbose);

        // class conditional probability
        inline arma::mat predict_prob(const arma::mat& beta,
                                      const arma::mat& x) const
        {
            return Abclass::predict_prob(x * beta);
        }

        // accuracy for tuning
        inline double accuracy(const arma::mat& beta,
                               const arma::mat& x,
                               const arma::uvec& y) const
        {
            return Abclass::accuracy(x * beta, y);
        }

        inline arma::vec gen_group_weight(
            const arma::vec& group_weight = arma::vec()
            ) const
        {
            if (group_weight.n_elem < p1_) {
                arma::vec out { arma::ones(p1_) };
                out[0] = 0.0;
                if (group_weight.is_empty()) {
                    return out;
                }
                if (group_weight.n_elem == p0_) {
                    for (size_t j {1}; j < p1_; ++j) {
                        out[j] = group_weight[j - 1];
                    }
                    return out;
                }
            } else if (group_weight.n_elem == p1_) {
                if (arma::any(group_weight < 0.0)) {
                    throw std::range_error(
                        "The 'group_weight' cannot be negative.");
                }
                return group_weight;
            }
            // else
            throw std::range_error("Incorrect length of the 'group_weight'.");
        }

        inline void set_group_weight(
            const arma::vec& group_weight = arma::vec()
            )
        {
            group_weight_ = gen_group_weight(group_weight);
        }

    };

    // run one GMD cycle over active sets
    inline void AbclassGroupLasso::run_one_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::uvec& is_active,
        const double lambda,
        const bool update_active,
        const unsigned int verbose
        )
    {
        double ell_verbose { 0.0 };
        if (verbose > 1) {
            Rcpp::Rcout << "\nStarting values of beta:\n";
            Rcpp::Rcout << beta << "\n";
            Rcpp::Rcout << "The active set of beta:\n";
            Rcpp::Rcout << arma2rvec(is_active) << "\n";
            ell_verbose = objective(inner, beta, lambda, group_weight_);
        };
        for (size_t j {0}; j < p1_; ++j) {
            if (is_active(j) == 0) {
                continue;
            }
            arma::rowvec old_beta_j { beta.row(j) };
            double gamma_j { gmd_lowerbound_(j) };
            arma::rowvec uj {
                - gmd_gradient(inner, j) + gamma_j * beta.row(j)
            };
            double wj { group_weight_(j) } ;
            double pos_part { 1 - lambda * wj / l2_norm(uj) };
            // update beta
            if (is_le(pos_part, 0.0)) {
                beta.row(j) = arma::zeros<arma::rowvec>(km1_);
            } else {
                beta.row(j) = uj * pos_part / gamma_j;
            }
            for (size_t i {0}; i < n_obs_; ++i) {
                inner(i) += x_(i, j) *
                    arma::accu((beta.row(j) - old_beta_j) % vertex_.row(y_(i)));
            }
            if (update_active) {
                // check if it has been shrinkaged to zero
                if (arma::any(beta.row(j) != 0.0)) {
                    is_active(j) = 1;
                } else {
                    is_active(j) = 0;
                }
            }
        }
        if (verbose > 1) {
            double ell_old { ell_verbose };
            Rcpp::Rcout << "The objective function changed\n";
            Rprintf("  from %15.15f\n", ell_verbose);
            ell_verbose = objective(inner, beta, lambda, group_weight_);
            Rprintf("    to %15.15f\n", ell_verbose);
            if (ell_verbose > ell_old) {
                Rcpp::Rcout << "Warning: "
                            << "the objective function somehow increased\n";
            }
        }
    }

    // run CMD cycles over active sets
    inline void AbclassGroupLasso::run_gmd_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::uvec& is_active,
        const double lambda,
        const bool varying_active_set,
        const unsigned int max_iter,
        const double rel_tol,
        const unsigned int verbose
        )
    {
        size_t i {0};
        arma::mat beta0 { beta };
        arma::uvec is_active_stored { is_active };
        // use active-set if p > n ("helps when p >> n")
        if (varying_active_set) {
            while (i < max_iter) {
                arma::uvec is_active_new { is_active };
                // cycles over the active set
                size_t ii {0};
                while (ii < max_iter) {
                    run_one_active_cycle(beta, inner, is_active_new,
                                         lambda, true, verbose);
                    if (rel_diff(beta0, beta) < rel_tol) {
                        num_iter_ = ii + 1;
                        break;
                    }
                    beta0 = beta;
                    ii++;
                }
                // run a full cycle over the converged beta
                run_one_active_cycle(beta, inner, is_active_stored,
                                     lambda, true, verbose);
                // check two active sets coincide
                if (is_gt(l1_norm(is_active_new - is_active_stored), 0)) {
                    // if different, repeat this process
                    if (verbose > 1) {
                        Rcpp::Rcout << "Enlarged the active set after "
                                    << num_iter_ + 1
                                    << " iteration(s)\n";
                    }
                    // recover the active set
                    is_active_stored = is_active;
                    i++;
                } else {
                    if (verbose > 1) {
                        Rcpp::Rcout << "Converged over the active set after "
                                    << num_iter_ + 1
                                    << " iteration(s)\n";
                    }
                    num_iter_ = i + 1;
                    break;
                }
            }
        } else {
            // regular coordinate descent
            while (i < max_iter) {
                run_one_active_cycle(beta, inner, is_active_stored,
                                     lambda, false, verbose);
                if (rel_diff(beta0, beta) < rel_tol) {
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
    inline void AbclassGroupLasso::fit(
        const arma::vec& lambda,
        const unsigned int nlambda,
        const double lambda_min_ratio,
        const arma::vec& group_weight,
        const unsigned int max_iter,
        const double rel_tol,
        const bool varying_active_set,
        const unsigned int verbose
        )
    {
        // set the CMD lowerbound
        set_gmd_lowerbound();
        // set group weight
        group_weight_ = gen_group_weight(group_weight);
        arma::uvec penalty_group { arma::find(group_weight_ > 0.0) };
        arma::uvec penalty_free { arma::find(group_weight_ <= 0.0) };
        // record control
        rel_tol_ = rel_tol;
        max_iter_ = max_iter;
        varying_active_set_ = varying_active_set;
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
            tmp /= group_weight_(*it);
            if (lambda_max_ < tmp) {
                lambda_max_ = tmp;
            }
        }
        // set up lambda sequence
        if (lambda.empty()) {
            double log_lambda_max { std::log(lambda_max_) };
            lambda_ = arma::exp(
                arma::linspace(log_lambda_max,
                               log_lambda_max + std::log(lambda_min_ratio),
                               nlambda)
                );
        } else {
            lambda_ = arma::reverse(arma::unique(lambda));
        }
        // initialize the estimate cube
        coef_ = arma::cube(p1_, km1_, lambda_.n_elem, arma::fill::zeros);

        double one_strong_rhs { 0.0 };
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::uvec is_active_strong { arma::zeros<arma::uvec>(p1_) };
        // only need to estimate beta not in the penalty group
        for (arma::uvec::iterator it { penalty_free.begin() };
             it != penalty_free.end(); ++it) {
            is_active_strong(*it) = 1;
        }
        run_gmd_active_cycle(one_beta, one_inner, is_active_strong,
                             lambda_max_, false,
                             max_iter, rel_tol, verbose);
        // optim with varying active set when p > n
        double old_lambda { lambda_max_ }; // for strong rule
        // main loop: for each lambda
        for (size_t li { 0 }; li < lambda_.n_elem; ++li) {
            double lambda_li { lambda_(li) };
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
                double one_strong_lhs { l2_norm(one_grad_beta.row(*it)) };
                one_strong_rhs = group_weight_(*it) *
                    (2 * lambda_li - old_lambda);
                if (one_strong_lhs >= one_strong_rhs) {
                    is_active_strong(*it) = 1;
                } else {
                    is_active_strong(*it) = 0;
                    one_beta.row(*it) = arma::zeros<arma::rowvec>(km1_);
                }
            }
            old_lambda = lambda_li;
            arma::uvec is_active_strong_new { is_active_strong };
            bool kkt_failed { true };
            one_strong_rhs = lambda_li;
            // eventually, strong rule will guess correctly
            while (kkt_failed) {
                // update beta
                run_gmd_active_cycle(one_beta, one_inner, is_active_strong,
                                     lambda_li, varying_active_set,
                                     max_iter, rel_tol, verbose);
                if (verbose > 0) {
                    msg("\nChecking the KKT condition for the null set.");
                }
                // check kkt condition
                for (arma::uvec::iterator it { penalty_group.begin() };
                     it != penalty_group.end(); ++it) {
                    if (is_active_strong(*it) > 0) {
                        continue;
                    }

                    if (l2_norm(gmd_gradient(one_inner, *it)) >
                        one_strong_rhs * group_weight_(*it)) {
                        // update active set
                        is_active_strong_new(*it) = 1;
                    }
                }
                if (l1_norm(is_active_strong_new - is_active_strong) > 0) {
                    if (verbose > 0) {
                        Rcpp::Rcout << "\nThe strong rule failed."
                                    << "\nOld active set:\n";
                        Rcpp::Rcout << is_active_strong << "\n";
                        Rcpp::Rcout << "\nNew active set:\n";
                        Rcpp::Rcout << is_active_strong_new << "\n";
                    }
                    is_active_strong = is_active_strong_new;
                } else {
                    if (verbose > 0) {
                        Rcpp::Rcout << "\nThe strong rule worked.\n";
                    }
                    kkt_failed = false;
                }
            }
            coef_.slice(li) = rescale_coef(one_beta);
        }
    }

}


#endif /* ABCLASS_ABCLASS_GROUP_LASSO_H */
