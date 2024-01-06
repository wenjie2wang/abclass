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

#ifndef ABCLASS_ABCLASS_NET_H
#define ABCLASS_ABCLASS_NET_H

#include <utility>
#include <RcppArmadillo.h>
#include "Abclass.h"
#include "Control.h"
#include "utils.h"

namespace abclass
{
    // the angle-based classifier with elastic-net penalty
    // estimation by coordinate-majorization-descent algorithm
    template <typename T_loss, typename T_x>
    class AbclassNet : public Abclass<T_loss, T_x>
    {
    protected:
        // data members
        using Abclass<T_loss, T_x>::inter_;
        using Abclass<T_loss, T_x>::km1_;
        using Abclass<T_loss, T_x>::mm_lowerbound_;
        using Abclass<T_loss, T_x>::mm_lowerbound0_;
        using Abclass<T_loss, T_x>::null_loss_;
        using Abclass<T_loss, T_x>::dn_obs_;

        // function members
        using Abclass<T_loss, T_x>::rescale_coef;
        using Abclass<T_loss, T_x>::get_vertex_y;
        using Abclass<T_loss, T_x>::objective0;
        using Abclass<T_loss, T_x>::loss_derivative;

        // common methods
        inline double regularization(const arma::mat& beta,
                                     const double l1_lambda,
                                     const double l2_lambda) const
        {
            if (control_.intercept_) {
                arma::mat beta0int { beta.tail_rows(p0_) };
                return l1_lambda * l1_norm(beta0int) +
                    0.5 * l2_lambda * l2_norm_square(beta0int);
            }
            return l1_lambda * l1_norm(beta) +
                0.5 * l2_lambda * l2_norm_square(beta);
        }

        // objective = loss / n + regularization
        inline double objective(const arma::vec& inner,
                                const arma::mat& beta,
                                const double l1_lambda,
                                const double l2_lambda) const
        {
            return objective0(inner) / dn_obs_ +
                regularization(beta, l1_lambda, l2_lambda);
        }

        // define gradient function at (l, j) for the given inner product
        inline double mm_gradient(const arma::vec& inner,
                                  const arma::vec& vj_xl) const
        {
            arma::vec inner_grad { loss_derivative(inner) };
            // avoid coefficients diverging
            inner_grad.elem(arma::find(inner_grad >
                                       control_.max_grad_)).zeros();
            return arma::mean(control_.obs_weight_ % vj_xl % inner_grad);
        }

        // gradient matrix regarding coef only (excluding intercept)
        // only use it to determine the large-enough lambda
        inline arma::mat gradient(const arma::vec& inner) const
        {
            arma::mat out { arma::zeros(p0_, km1_) };
            arma::vec inner_grad { loss_derivative(inner) };
            for (size_t j {0}; j < km1_; ++j) {
                const arma::vec w_v_j {
                    control_.obs_weight_ % get_vertex_y(j)
                };
                for (size_t l {0}; l < p0_; ++l) {
                    const arma::vec w_vj_xl { w_v_j % x_.col(l) };
                    out(l, j) = arma::mean(w_vj_xl % inner_grad);
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
        inline void run_cmd_active_cycles(arma::mat& beta,
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
        inline void run_cmd_full_cycles(arma::mat& beta,
                                        arma::vec& inner,
                                        const double l1_lambda,
                                        const double l2_lambda,
                                        const unsigned int max_iter,
                                        const double epsilon,
                                        const unsigned int verbose);

    public:

        // inherit constructors
        using Abclass<T_loss, T_x>::Abclass;

        // specifics for template inheritance
        using Abclass<T_loss, T_x>::control_;
        using Abclass<T_loss, T_x>::x_;
        using Abclass<T_loss, T_x>::p0_;
        using Abclass<T_loss, T_x>::p1_;
        using Abclass<T_loss, T_x>::n_obs_;
        using Abclass<T_loss, T_x>::ex_vertex_;
        using Abclass<T_loss, T_x>::et_npermuted_;

        using Abclass<T_loss, T_x>::coef_;
        using Abclass<T_loss, T_x>::loss_;
        using Abclass<T_loss, T_x>::penalty_;
        using Abclass<T_loss, T_x>::objective_;

        using Abclass<T_loss, T_x>::set_mm_lowerbound;

        // regularization
        // the "big" enough lambda => zero coef unless alpha = 0
        double l1_lambda_max_;
        double lambda_max_;
        // did user specified a customized lambda sequence?
        bool custom_lambda_ = false;

        // for a sequence of lambda's
        inline void fit();

    };

    // run one CMD cycle over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassNet<T_loss, T_x>::run_one_active_cycle(
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
            Rcpp::Rcout << "\nStarting values of beta:\n";
            Rcpp::Rcout << beta << "\n";
            Rcpp::Rcout << "The active set of beta:\n";
            Rcpp::Rcout << is_active << "\n";
        };
        for (size_t j {0}; j < km1_; ++j) {
            arma::vec v_j { get_vertex_y(j) };
            // intercept
            if (control_.intercept_) {
                double dlj { mm_gradient(inner, v_j) };
                double tmp_delta { - dlj / mm_lowerbound0_ };
                beta(0, j) += tmp_delta;
                inner += tmp_delta * v_j;
            }
            for (size_t l { 0 }; l < p0_; ++l) {
                if (is_active(l, j) == 0) {
                    continue;
                }
                size_t l1 { l + inter_ };
                arma::vec vj_xl { x_.col(l) % v_j };
                double dlj { mm_gradient(inner, vj_xl) };
                double tmp { beta(l1, j) };
                // if mm_lowerbound = 0 and l1_lambda > 0, numer will be 0
                double numer {
                    soft_threshold(mm_lowerbound_(l) * beta(l1, j) - dlj,
                                   l1_lambda)
                };
                // update beta
                if (isAlmostEqual(numer, 0)) {
                    beta(l1, j) = 0;
                } else {
                    double denom { mm_lowerbound_(l) + l2_lambda };
                    beta(l1, j) = numer / denom;
                }
                inner += (beta(l1, j) - tmp) * vj_xl;
                if (update_active) {
                    // check if it has been shrinkaged to zero
                    if (isAlmostEqual(beta(l1, j), 0)) {
                        is_active(l, j) = 0;
                    } else {
                        is_active(l, j) = 1;
                    }
                }
            }
        }
    }

    // run CMD cycles over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassNet<T_loss, T_x>::run_cmd_active_cycles(
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
        size_t i {0}, num_iter {0};
        // arma::mat beta0 { beta };
        double obj0 { objective(inner, beta, l1_lambda, l2_lambda) },
            obj1 { obj0 };
        // use active-set if p > n ("helps when p >> n")
        if (varying_active_set) {
            arma::umat is_active_strong { is_active },
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
                                         l1_lambda, l2_lambda, true, verbose);
                    // if (rel_diff(beta0, beta) < epsilon) {
                    //     break;
                    // }
                    // beta0 = beta;
                    obj1 = objective(inner, beta, l1_lambda, l2_lambda);
                    // optional: throw warning if objective function increases
                    if (verbose > 1) {
                        Rcpp::Rcout << "The objective function changed\n";
                        Rprintf("  from %15.15f\n", obj0);
                        Rprintf("    to %15.15f\n", obj1);
                        if (obj0 > obj1) {
                            Rcpp::Rcout << "Warning: "
                                        << "the function objective "
                                        << "somehow increased.\n";
                        }
                    }
                    if (std::abs(obj1 - obj0) < epsilon) {
                        break;
                    }
                    obj0 = obj1;
                    ii++;
                }
                // run a full cycle over the converged beta
                run_one_active_cycle(beta, inner, is_active,
                                     l1_lambda, l2_lambda, true, verbose);
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
                                    << num_iter + 1
                                    << " iteration(s)\n";
                    }
                    is_active_varying = is_active;
                    // recover the active set
                    is_active = is_active_strong;
                    i++;
                } else {
                    break;
                }
            }
        } else {
            // regular coordinate descent
            while (i < max_iter) {
                Rcpp::checkUserInterrupt();
                num_iter = i + 1;
                run_one_active_cycle(beta, inner, is_active,
                                     l1_lambda, l2_lambda, false, verbose);
                // if (rel_diff(beta0, beta) < epsilon) {
                //     break;
                // }
                // beta0 = beta;
                obj1 = objective(inner, beta, l1_lambda, l2_lambda);
                // optional: throw warning if objective function increases
                if (verbose > 1) {
                    Rcpp::Rcout << "The objective function changed\n";
                    Rprintf("  from %15.15f\n", obj0);
                    Rprintf("    to %15.15f\n", obj1);
                    if (obj0 > obj1) {
                        Rcpp::Rcout << "Warning: "
                                    << "the function objective "
                                    << "somehow increased.\n";
                    }
                }
                if (std::abs(obj1 - obj0) < epsilon) {
                    break;
                }
                obj0 = obj1;
                i++;
            }
        }
        if (verbose > 0) {
            if (num_iter < max_iter) {
                Rcpp::Rcout << "Outer loop converged over the active set after "
                            << num_iter
                            << " iteration(s)\n";
                Rcpp::Rcout << "The size of active set is "
                            << l1_norm(is_active) << ".\n";
            } else {
                msg("Outer loop reached the maximum number of iteratons.");
            }
        }
    }

    // one full cycle for coordinate-descent
    template <typename T_loss, typename T_x>
    inline void AbclassNet<T_loss, T_x>::run_one_full_cycle(
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
        for (size_t j {0}; j < km1_; ++j) {
            arma::vec v_j { get_vertex_y(j) };
            // intercept
            if (control_.intercept_) {
                double dlj { mm_gradient(inner, v_j) };
                double tmp_delta { - dlj / mm_lowerbound0_ };
                beta(0, j) += tmp_delta;
                inner += tmp_delta * v_j;
            }
            // others
            for (size_t l { 0 }; l < p0_; ++l) {
                size_t l1 { l + inter_ };
                arma::vec vj_xl { x_.col(l) % v_j };
                double dlj { mm_gradient(inner, vj_xl) };
                double tmp { beta(l1, j) };
                // if cmd_lowerbound = 0 and l1_lambda > 0, numer will be 0
                double numer {
                    soft_threshold(mm_lowerbound_(l) * tmp - dlj,
                                   l1_lambda)
                };
                // update beta
                if (isAlmostEqual(numer, 0)) {
                    beta(l1, j) = 0;
                } else {
                    double denom { mm_lowerbound_(l) + l2_lambda };
                    beta(l1, j) = numer / denom;
                }
                inner += (beta(l1, j) - tmp) * vj_xl;
            }
        }
    }

    // run full cycles till convergence or reach max number of iterations
    template <typename T_loss, typename T_x>
    inline void AbclassNet<T_loss, T_x>::run_cmd_full_cycles(
        arma::mat& beta,
        arma::vec& inner,
        const double l1_lambda,
        const double l2_lambda,
        const unsigned int max_iter,
        const double epsilon,
        const unsigned int verbose
        )
    {
        // arma::mat beta0 { beta };
        double obj0 { objective(inner, beta, l1_lambda, l2_lambda) },
            obj1 { obj0 };
        size_t num_iter {0};
        for (size_t i {0}; i < max_iter; ++i) {
            Rcpp::checkUserInterrupt();
            num_iter = i + 1;
            run_one_full_cycle(beta, inner, l1_lambda, l2_lambda, verbose);
            // if (rel_diff(beta0, beta) < epsilon) {
            //     break;
            // }
            // beta0 = beta;
            obj1 = objective(inner, beta, l1_lambda, l2_lambda);
            // optional: throw warning if objective function increases
            if (verbose > 1) {
                Rcpp::Rcout << "The objective function changed\n";
                Rprintf("  from %15.15f\n", obj0);
                Rprintf("    to %15.15f\n", obj1);
                if (obj0 > obj1) {
                    Rcpp::Rcout << "Warning: "
                                << "the function objective "
                                << "somehow increased.\n";
                }
            }
            if (std::abs(obj1 - obj0) < epsilon) {
                break;
            }
            obj0 = obj1;
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

    // for a sequence of lambda's
    // lambda * (alpha * lasso + (1 - alpha) / 2 * ridge)
    template <typename T_loss, typename T_x>
    inline void AbclassNet<T_loss, T_x>::fit()
    {
        // set the CMD lowerbound
        set_mm_lowerbound();
        // penalty for covariates with positive penalty factors only
        arma::uvec penalty_group { arma::find(control_.penalty_factor_ > 0.0) };
        // initialize
        arma::vec one_inner;
        if (control_.has_offset_) {
            arma::mat ex_inner { ex_vertex_ % control_.offset_ };
            one_inner = arma::sum(ex_inner, 1);
        } else {
            one_inner = arma::zeros(n_obs_);
        }
        arma::mat one_beta { arma::zeros(p1_, km1_) },
            one_grad_beta { one_beta };
        const bool is_ridge_only { isAlmostEqual(control_.alpha_, 0.0) };
        double l1_lambda { 0.0 }, l2_lambda { 0.0 };
        if (! control_.lambda_.empty()) {
            control_.lambda_ = arma::reverse(arma::unique(control_.lambda_));
            control_.nlambda_ = control_.lambda_.n_elem;
            custom_lambda_ = true;
        }
        // if alpha = 0 and customized lambda
        if (is_ridge_only && custom_lambda_) {
            l1_lambda_max_ = - 1.0; // not well defined
            lambda_max_ = - 1.0;    // not well defined
        } else {
            // need to determine lambda_max
            one_grad_beta = arma::abs(gradient(one_inner));
            l1_lambda_max_ = one_grad_beta.max();
            lambda_max_ =  l1_lambda_max_ / std::max(control_.alpha_, 1e-2);
            // set up lambda sequence
            if (! custom_lambda_) {
                double log_lambda_max { std::log(lambda_max_) };
                control_.lambda_ = arma::exp(
                    arma::linspace(log_lambda_max,
                                   log_lambda_max +
                                   std::log(control_.lambda_min_ratio_),
                                   control_.nlambda_)
                    );
            }
        }
        // initialize the estimate cube
        coef_ = arma::cube(p1_, km1_, control_.lambda_.n_elem,
                           arma::fill::zeros);
        objective_ = penalty_ = loss_ = arma::zeros(control_.lambda_.n_elem);
        // set epsilon from the default null objective, n
        null_loss_ = dn_obs_;
        double epsilon0 { control_.epsilon_ };
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::umat is_active_strong { arma::zeros<arma::umat>(p0_, km1_) };
        if (control_.intercept_) {
            // only need to estimate intercept
            run_cmd_active_cycles(one_beta,
                                  one_inner,
                                  is_active_strong,
                                  0, // does not matter
                                  0, // does not matter
                                  false,
                                  control_.max_iter_,
                                  epsilon0,
                                  control_.verbose_);
            // update epsilon0
            null_loss_ = objective0(one_inner);
            epsilon0 = exp_log_sum(control_.epsilon_, null_loss_ / dn_obs_);
        }
        // for pure ridge penalty
        if (is_ridge_only) {
            for (size_t li { 0 }; li < control_.lambda_.n_elem; ++li) {
                l2_lambda = control_.lambda_(li);
                run_cmd_full_cycles(one_beta,
                                    one_inner,
                                    l1_lambda,
                                    l2_lambda,
                                    control_.max_iter_,
                                    epsilon0,
                                    control_.verbose_);
                coef_.slice(li) = rescale_coef(one_beta);
                loss_(li) = objective0(one_inner);
                penalty_(li) = regularization(one_beta, l1_lambda, l2_lambda);
                objective_(li) = loss_(li) / dn_obs_ + penalty_(li);
            }
            return;             // early exit
        }
        // else, not just ridge penalty with l1_lambda > 0
        // for strong rule
        double one_strong_rhs { 0.0 }, old_l1_lambda { l1_lambda_max_ };
        // main loop: for each lambda
        for (size_t li { 0 }; li < control_.lambda_.n_elem; ++li) {
            double lambda_li { control_.lambda_(li) };
            l1_lambda = lambda_li * control_.alpha_;
            l2_lambda = lambda_li * (1 - control_.alpha_);
            // early exit for lambda greater than lambda_max_
            // note that lambda is sorted
            if (l1_lambda >= l1_lambda_max_) {
                coef_.slice(li) = rescale_coef(one_beta);
                loss_(li) = null_loss_;
                penalty_(li) = regularization(one_beta, l1_lambda, l2_lambda);
                objective_(li) = loss_(li) / dn_obs_ + penalty_(li);
                continue;
            }
            // update active set by strong rule
            one_grad_beta = arma::abs(gradient(one_inner));
            one_strong_rhs = 2 * l1_lambda - old_l1_lambda;
            old_l1_lambda = l1_lambda;
            for (size_t j { 0 }; j < km1_; ++j) {
                for (size_t l { 0 }; l < p0_; ++l) {
                    if (is_active_strong(l, j) > 0) {
                        continue;
                    }
                    if (one_grad_beta(l, j) >= one_strong_rhs) {
                        is_active_strong(l, j) = 1;
                    }
                }
            }
            bool kkt_failed { true };
            // eventually, strong rule will guess correctly
            while (kkt_failed) {
                arma::umat is_active_strong_old { is_active_strong };
                arma::umat is_strong_rule_failed {
                    arma::zeros<arma::umat>(arma::size(is_active_strong))
                };
                // update beta
                run_cmd_active_cycles(one_beta,
                                      one_inner,
                                      is_active_strong,
                                      l1_lambda,
                                      l2_lambda,
                                      control_.varying_active_set_,
                                      control_.max_iter_,
                                      epsilon0,
                                      control_.verbose_);
                if (control_.verbose_ > 0) {
                    msg("Checking the KKT condition for the null set.");
                }
                // check kkt condition
                const arma::vec inner_grad {
                    control_.obs_weight_ % loss_derivative(one_inner)
                };
                for (size_t l { 0 }; l < p0_; ++l) {
                    arma::vec x_l { x_.col(l) };
                    for (size_t j { 0 }; j < km1_; ++j) {
                        if (is_active_strong_old(l, j) > 0) {
                            continue;
                        }
                        arma::vec vj_xl { x_l % get_vertex_y(j) };
                        double tmp { arma::mean(vj_xl % inner_grad) };
                        if (std::abs(tmp) > l1_lambda) {
                            // update active set
                            is_strong_rule_failed(l, j) = 1;
                        }
                    }
                }
                if (arma::accu(is_strong_rule_failed) > 0) {
                    is_active_strong = is_active_strong_old ||
                        is_strong_rule_failed;
                    if (control_.verbose_ > 0) {
                        Rcpp::Rcout << "The strong rule failed.\n"
                                    << "The size of old active set: "
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
                            "the largest lamabda specified; ",
                            "the returned solution may not be sensible.\n",
                            "Suggestion: increase 'lambda', ",
                            "'lambda_min_ratio' or 'nlambda'?\n");
                        coef_ = coef_.head_slices(1);
                        loss_ = loss_(0);
                        penalty_ = penalty_(0);
                        objective_(0) = loss_(0) / dn_obs_ + penalty_(0);
                    } else {
                        coef_ = coef_.head_slices(li);
                        loss_ = loss_.head(li);
                        penalty_ = penalty_.head(li);
                        objective_ = objective_.head(li);
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
            loss_(li) = objective0(one_inner);
            penalty_(li) = regularization(one_beta, l1_lambda, l2_lambda);
            objective_(li) = loss_(li) / dn_obs_ + penalty_(li);
        }
    }

}  // abclass


#endif
