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

#ifndef ABCLASS_ABCLASS_NET_H
#define ABCLASS_ABCLASS_NET_H

#include <utility>
#include <RcppArmadillo.h>
#include "Abclass.h"
#include "utils.h"

namespace abclass
{

    // the angle-based classifier with elastic-net penalty
    // estimation by coordinate-majorization-descent algorithm
    class AbclassNet : public Abclass
    {
    protected:

        // for regularized coordinate majorization descent
        arma::rowvec cmd_lowerbound_; // 1 by p1_

        // pure virtual functions
        virtual void set_cmd_lowerbound() = 0;
        virtual double objective0(const arma::vec& inner) const = 0;

        // common methods
        inline double regularization(const arma::mat& beta,
                                     const double l1_lambda,
                                     const double l2_lambda) const
        {
            if (intercept_) {
                arma::mat beta0int { beta.tail_rows(p0_) };
                return l1_lambda * l1_norm(beta0int) +
                    l2_lambda * l2_norm_square(beta0int);
            }
            return l1_lambda * l1_norm(beta) +
                l2_lambda * l2_norm_square(beta);
        }

        // objective function with regularization
        inline double objective(const arma::vec& inner,
                                const arma::mat& beta,
                                const double l1_lambda,
                                const double l2_lambda) const
        {
            return objective0(inner) +
                regularization(beta, l1_lambda, l2_lambda);
        }

        // define gradient function at (l, j) for the given inner product
        inline double cmd_gradient(const arma::vec& inner,
                                   const arma::vec& vj_xl) const
        {
            arma::vec inner_grad { loss_derivative(inner) };
            return arma::mean(obs_weight_ % vj_xl % inner_grad);
        }

        // gradient matrix
        inline arma::mat gradient(const arma::vec& inner) const
        {
            arma::mat out { arma::zeros(p1_, km1_) };
            arma::vec inner_grad { loss_derivative(inner) };
            for (size_t j {0}; j < km1_; ++j) {
                const arma::vec w_v_j { obs_weight_ % get_vertex_y(j) };
                for (size_t l {0}; l < p1_; ++l) {
                    const arma::vec w_vj_xl { w_v_j % x_.col(l) };
                    out(l, j) = arma::mean(w_vj_xl % inner_grad);
                }
            }
            return out;
        }

        inline double max_diff(const arma::mat& beta_new,
                               const arma::mat& beta_old) const
        {
            double out { 0.0 };
            for (size_t j {0}; j < km1_; ++j) {
                for (size_t i {0}; i < p1_; ++i) {
                    double tmp {
                        cmd_lowerbound_(i) *
                        std::pow(beta_new(i, j) - beta_old(i, j), 2)
                    };
                    if (out < tmp) {
                        out = tmp;
                    }
                }
            }
            return out;
        }

        // run one cycle of coordinate descent over a given active set
        inline void run_one_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::umat& is_active,
                                         const double l1_lambda,
                                         const double l2_lambda,
                                         const bool update_active,
                                         const unsigned int verbose);

        // run complete cycles of CMD for a given active set and given lambda's
        inline void run_cmd_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::umat& is_active,
                                         const double l1_lambda,
                                         const double l2_lambda,
                                         const bool varying_active_set,
                                         const unsigned int max_iter,
                                         const double epsilon,
                                         const unsigned int verbose);

        // one full cycle for coordinate-descent
        inline void run_one_full_cycle(arma::mat& beta,
                                       arma::vec& inner,
                                       const double l1_lambda,
                                       const double l2_lambda,
                                       const unsigned int verbose);

        // run full cycles of CMD for given lambda's
        inline void run_cmd_full_cycle(arma::mat& beta,
                                       arma::vec& inner,
                                       const double l1_lambda,
                                       const double l2_lambda,
                                       const unsigned int max_iter,
                                       const double epsilon,
                                       const unsigned int verbose);

    public:

        // inherit constructors
        using Abclass::Abclass;

        // regularization
        // the "big" enough lambda => zero coef unless alpha = 0
        double l1_lambda_max_;
        double lambda_max_;
        double alpha_;            // [0, 1]
        arma::vec lambda_;        // lambda sequence
        // did user specified a customized lambda sequence?
        bool custom_lambda_ = false;
        double lambda_min_ratio_ = 0.01;

        // estimates
        arma::cube coef_;         // p1_ by km1_

        // tuning by cross-validation
        arma::mat cv_accuracy_;
        arma::vec cv_accuracy_mean_;
        arma::vec cv_accuracy_sd_;

        // control
        double epsilon_;          // relative tolerance for convergence check
        unsigned int max_iter_;   // maximum number of iterations
        bool varying_active_set_; // if active set should be adaptive

        // cache
        unsigned int num_iter_;   // number of CMD cycles till convergence

        // for a sequence of lambda's
        inline void fit(const arma::vec& lambda,
                        const double alpha,
                        const unsigned int nlambda,
                        const double lambda_min_ratio,
                        const unsigned int max_iter,
                        const double epsilon,
                        const bool varying_active_set,
                        const unsigned int verbose);

        // class conditional probability
        inline arma::mat predict_prob(const arma::mat& beta,
                                      const arma::mat& x) const
        {
            return Abclass::predict_prob(x * beta);
        }
        // prediction based on the inner products
        inline arma::uvec predict_y(const arma::mat& beta,
                                    const arma::mat& x) const
        {
            return Abclass::predict_y(x * beta);
        }
        // accuracy for tuning
        inline double accuracy(const arma::mat& beta,
                               const arma::mat& x,
                               const arma::uvec& y) const
        {
            return Abclass::accuracy(x * beta, y);
        }

    };

    // run one CMD cycle over active sets
    inline void AbclassNet::run_one_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
        const bool update_active,
        const unsigned int verbose
        )
    {
        double ell_verbose { 0.0 };
        if (verbose > 2) {
            Rcpp::Rcout << "\nStarting values of beta:\n";
            Rcpp::Rcout << beta << "\n";
            Rcpp::Rcout << "The active set of beta:\n";
            Rcpp::Rcout << is_active << "\n";
        };
        if (verbose > 1) {
            ell_verbose = objective(inner, beta, l1_lambda, l2_lambda);
        }
        for (size_t j {0}; j < km1_; ++j) {
            arma::vec v_j { get_vertex_y(j) };
            for (size_t l {0}; l < p1_; ++l) {
                if (is_active(l, j) == 0) {
                    continue;
                }
                arma::vec vj_xl { x_.col(l) % v_j };
                double dlj { cmd_gradient(inner, vj_xl) };
                double tmp { beta(l, j) };
                // if cmd_lowerbound = 0 and l1_lambda > 0, numer will be 0
                double numer {
                    soft_threshold(
                        cmd_lowerbound_(l) * beta(l, j) - dlj,
                        l1_lambda * static_cast<double>(l >= int_intercept_)
                        )
                };
                // update beta
                if (isAlmostEqual(numer, 0)) {
                    beta(l, j) = 0;
                } else {
                    double denom { cmd_lowerbound_(l) + 2 * l2_lambda *
                        static_cast<double>(l >= int_intercept_)
                    };
                    beta(l, j) = numer / denom;
                }
                inner += (beta(l, j) - tmp) * vj_xl;
                if (update_active) {
                    // check if it has been shrinkaged to zero
                    if (isAlmostEqual(beta(l, j), 0)) {
                        is_active(l, j) = 0;
                    } else {
                        is_active(l, j) = 1;
                    }
                }
            }
        }
        if (verbose > 1) {
            double ell_old { ell_verbose };
            Rcpp::Rcout << "The objective function changed\n";
            Rprintf("  from %15.15f\n", ell_verbose);
            ell_verbose = objective(inner, beta, l1_lambda, l2_lambda);
            Rprintf("    to %15.15f\n", ell_verbose);
            if (ell_verbose > ell_old) {
                Rcpp::Rcout << "Warning: "
                            << "the objective function somehow increased\n";
            }
        }
    }

    // run CMD cycles over active sets
    inline void AbclassNet::run_cmd_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
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
            arma::umat is_active_strong { is_active },
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
                                         l1_lambda, l2_lambda, true, verbose);
                    if (rel_diff(beta0, beta) < epsilon) {
                        num_iter_ = ii + 1;
                        break;
                    }
                    beta0 = beta;
                    ii++;
                }
                // run a full cycle over the converged beta
                run_one_active_cycle(beta, inner, is_active,
                                     l1_lambda, l2_lambda, true, verbose);
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
                    }
                    num_iter_ = i + 1;
                    break;
                }
            }
        } else {
            // regular coordinate descent
            while (i < max_iter) {
                run_one_active_cycle(beta, inner, is_active,
                                     l1_lambda, l2_lambda, false, verbose);
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

    // one full cycle for coordinate-descent
    inline void AbclassNet::run_one_full_cycle(
        arma::mat& beta,
        arma::vec& inner,
        const double l1_lambda,
        const double l2_lambda,
        const unsigned int verbose
        )
    {
        double ell_verbose { 0.0 };
        if (verbose > 2) {
            Rcpp::Rcout << "\nStarting values of beta:\n";
            Rcpp::Rcout << beta << "\n";
        };
        if (verbose > 1) {
            ell_verbose = objective(inner, beta, l1_lambda, l2_lambda);
        }
        for (size_t j {0}; j < km1_; ++j) {
            arma::vec v_j { get_vertex_y(j) };
            for (size_t l {0}; l < p1_; ++l) {
                arma::vec vj_xl { v_j % x_.col(l) };
                double dlj { cmd_gradient(inner, vj_xl) };
                double tmp { beta(l, j) };
                // if cmd_lowerbound = 0 and l1_lambda > 0, numer will be 0
                double numer {
                    soft_threshold(
                        cmd_lowerbound_(l) * tmp - dlj,
                        l1_lambda * static_cast<double>(l >= int_intercept_)
                        )
                };
                // update beta
                if (isAlmostEqual(numer, 0)) {
                    beta(l, j) = 0;
                } else {
                    double denom {
                        cmd_lowerbound_(l) + 2 * l2_lambda *
                        static_cast<double>(l >= int_intercept_)
                    };
                    beta(l, j) = numer / denom;
                }
                inner += (beta(l, j) - tmp) * vj_xl;
            }
        }
        if (verbose > 1) {
            double ell_old { ell_verbose };
            Rcpp::Rcout << "The objective function changed\n";
            Rprintf("  from %15.15f\n", ell_verbose);
            ell_verbose = objective(inner, beta, l1_lambda, l2_lambda);
            Rprintf("    to %15.15f\n", ell_verbose);
            if (ell_verbose > ell_old) {
                Rcpp::Rcout << "Warning: "
                            << "the objective function somehow increased\n";
            }
        }
    }

    // run full cycles till convergence or reach max number of iterations
    inline void AbclassNet::run_cmd_full_cycle(
        arma::mat& beta,
        arma::vec& inner,
        const double l1_lambda,
        const double l2_lambda,
        const unsigned int max_iter,
        const double epsilon,
        const unsigned int verbose
        )
    {
        arma::mat beta0 { beta };
        for (size_t i {0}; i < max_iter; ++i) {
            run_one_full_cycle(beta, inner, l1_lambda, l2_lambda, verbose);
            if (rel_diff(beta0, beta) < epsilon) {
                num_iter_ = i + 1;
                break;
            }
            beta0 = beta;
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
    // lambda * (alpha * lasso + (1 - alpha) / 2 * ridge)
    inline void AbclassNet::fit(
        const arma::vec& lambda,
        const double alpha,
        const unsigned int nlambda,
        const double lambda_min_ratio,
        const unsigned int max_iter,
        const double epsilon,
        const bool varying_active_set,
        const unsigned int verbose
        )
    {
        // set the CMD lowerbound
        set_cmd_lowerbound();
        // check alpha
        if ((alpha < 0) || (alpha > 1)) {
            throw std::range_error("The 'alpha' must be between 0 and 1.");
        }
        alpha_ = alpha;
        // record control
        epsilon_ = epsilon;
        max_iter_ = max_iter;
        varying_active_set_ = varying_active_set;
        // initialize
        arma::vec one_inner { arma::zeros(n_obs_) };
        arma::mat one_beta { arma::zeros(p1_, km1_) },
            one_grad_beta { one_beta };
        const bool is_ridge_only { isAlmostEqual(alpha, 0.0) };
        double l1_lambda, l2_lambda;
        // if alpha = 0 and lambda is specified
        if (is_ridge_only && ! lambda.empty()) {
            lambda_ = arma::reverse(arma::unique(lambda));
            custom_lambda_ = true;
            l1_lambda_max_ = - 1.0; // not well defined
            lambda_max_ = - 1.0;    // not well defined
        } else {
            // need to determine lambda_max
            one_grad_beta = arma::abs(gradient(one_inner));
            // large enough lambda for all-zero coef (except intercept terms)
            l1_lambda_max_ = one_grad_beta.tail_rows(p0_).max();
            lambda_max_ =  l1_lambda_max_ / std::max(alpha, 1e-2);
            // set up lambda sequence
            if (lambda.empty()) {
                double log_lambda_max { std::log(lambda_max_) };
                lambda_ = arma::exp(
                    arma::linspace(log_lambda_max,
                                   log_lambda_max + std::log(lambda_min_ratio),
                                   nlambda)
                    );
                lambda_min_ratio_ = lambda_min_ratio;
            } else {
                lambda_ = arma::reverse(arma::unique(lambda));
                custom_lambda_ = true;
            }
        }
        // initialize the estimate cube
        coef_ = arma::cube(p1_, km1_, lambda_.n_elem);
        // for ridge penalty
        if (is_ridge_only) {
            for (size_t li { 0 }; li < lambda_.n_elem; ++li) {
                run_cmd_full_cycle(one_beta, one_inner,
                                   0.0, 0.5 * lambda_(li),
                                   max_iter, epsilon, verbose);
                coef_.slice(li) = rescale_coef(one_beta);
            }
            return;             // early exit
        }
        // else, not just ridge penalty with l1_lambda > 0
        double one_strong_rhs { 0.0 };
        l2_lambda = 0.5 * lambda_max_ * (1 - alpha);
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::umat is_active_strong { arma::zeros<arma::umat>(p1_, km1_) };
        if (intercept_) {
            // only need to estimate intercept
            is_active_strong.row(0) = arma::ones<arma::umat>(1, km1_);
            run_cmd_active_cycle(one_beta, one_inner, is_active_strong,
                                 l1_lambda_max_, l2_lambda,
                                 false, max_iter, epsilon, verbose);
        }
        // optim with varying active set when p > n
        double old_l1_lambda { l1_lambda_max_ }; // for strong rule
        // main loop: for each lambda
        for (size_t li { 0 }; li < lambda_.n_elem; ++li) {
            double lambda_li { lambda_(li) };
            l1_lambda = lambda_li * alpha;
            l2_lambda = 0.5 * lambda_li * (1 - alpha);
            // early exit for lambda greater than lambda_max_
            // note that lambda is sorted
            if (l1_lambda >= l1_lambda_max_) {
                coef_.slice(li) = rescale_coef(one_beta);
                continue;
            }
            // update active set by strong rule
            one_grad_beta = arma::abs(gradient(one_inner));
            one_strong_rhs = 2 * l1_lambda - old_l1_lambda;
            old_l1_lambda = l1_lambda;
            for (size_t j { 0 }; j < km1_; ++j) {
                for (size_t l { int_intercept_ }; l < p1_; ++l) {
                    if (is_active_strong(l, j) > 0) {
                        continue;
                    }
                    if (one_grad_beta(l, j) >= one_strong_rhs) {
                        is_active_strong(l, j) = 1;
                    }
                }
            }
            bool kkt_failed { true };
            one_strong_rhs = l1_lambda;
            // eventually, strong rule will guess correctly
            while (kkt_failed) {
                arma::umat is_active_strong_old { is_active_strong };
                arma::umat is_strong_rule_failed {
                    arma::zeros<arma::umat>(arma::size(is_active_strong))
                };
                // update beta
                run_cmd_active_cycle(one_beta, one_inner, is_active_strong,
                                     l1_lambda, l2_lambda, varying_active_set,
                                     max_iter, epsilon, verbose);
                if (verbose > 0) {
                    msg("Checking the KKT condition for the null set.");
                }
                // check kkt condition
                for (size_t j { 0 }; j < km1_; ++j) {
                    arma::vec v_j { get_vertex_y(j) };
                    for (size_t l { int_intercept_ }; l < p1_; ++l) {
                        if (is_active_strong_old(l, j) > 0) {
                            continue;
                        }
                        arma::vec vj_xl { v_j % x_.col(l) };
                        if (std::abs(cmd_gradient(one_inner, vj_xl)) >
                            one_strong_rhs) {
                            // update active set
                            is_strong_rule_failed(l, j) = 1;
                        }
                    }
                }
                if (arma::accu(is_strong_rule_failed) > 0) {
                    is_active_strong = is_active_strong_old ||
                        is_strong_rule_failed;
                    if (verbose > 0) {
                        Rcpp::Rcout << "The strong rule failed.\n"
                                    << "The size of old active set: "
                                    << l1_norm(is_active_strong_old)
                                    << "\nThe size of new active set: "
                                    << l1_norm(is_active_strong)
                                    << "\n";
                    }
                } else {
                    if (verbose > 0) {
                        msg("The strong rule worked.\n");
                    }
                    kkt_failed = false;
                }
            }
            coef_.slice(li) = rescale_coef(one_beta);
        }
    }

}  // abclass


#endif
