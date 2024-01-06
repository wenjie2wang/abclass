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

#ifndef ABCLASS_ABCLASS_GROUP_H
#define ABCLASS_ABCLASS_GROUP_H

#include <RcppArmadillo.h>
#include "Abclass.h"
#include "Control.h"

namespace abclass
{
    // base class for the angle-based large margin classifiers
    // with group-wise regularization

    // T_x is intended to be arma::mat or arma::sp_mat
    template <typename T_loss, typename T_x>
    class AbclassGroup : public Abclass<T_loss, T_x>
    {
    protected:
        // data
        using Abclass<T_loss, T_x>::dn_obs_;
        using Abclass<T_loss, T_x>::km1_;
        using Abclass<T_loss, T_x>::inter_;
        using Abclass<T_loss, T_x>::mm_lowerbound_;
        using Abclass<T_loss, T_x>::mm_lowerbound0_;
        using Abclass<T_loss, T_x>::null_loss_;

        // functions
        using Abclass<T_loss, T_x>::loss_derivative;
        using Abclass<T_loss, T_x>::gen_penalty_factor;
        using Abclass<T_loss, T_x>::objective0;
        using Abclass<T_loss, T_x>::set_mm_lowerbound;

        // define gradient function for j-th predictor
        inline arma::rowvec mm_gradient(const arma::vec& inner,
                                        const unsigned int j) const
        {
            arma::vec inner_grad { loss_derivative(inner) };
            // avoid coefficients diverging
            inner_grad.elem(arma::find(inner_grad >
                                       control_.max_grad_)).zeros();
            arma::vec tmp_vec {
                x_.col(j) % control_.obs_weight_ % inner_grad
            };
            arma::rowvec out { tmp_vec.t() * ex_vertex_ };
            return out / dn_obs_;
        }
        // for intercept
        inline arma::rowvec mm_gradient0(const arma::vec& inner) const
        {
            arma::vec inner_grad { loss_derivative(inner) };
            arma::vec tmp_vec { control_.obs_weight_ % inner_grad };
            arma::rowvec out { tmp_vec.t() * ex_vertex_ };
            return out / dn_obs_;
        }

        // gradient matrix for beta
        inline arma::mat gradient(const arma::vec& inner) const
        {
            arma::mat out { arma::zeros(p0_, km1_) };
            arma::vec inner_grad { loss_derivative(inner) };
            for (size_t j {0}; j < p0_; ++j) {
                arma::vec tmp_vec {
                    x_.col(j) % control_.obs_weight_ % inner_grad
                };
                arma::rowvec tmp { tmp_vec.t() * ex_vertex_ };
                out.row(j) = tmp;
            }
            return out / dn_obs_;
        }

        // penalty function for beta >= 0 (default: lasso)
        inline virtual double penalty0(const double beta,
                                       const double l1_lambda,
                                       const double l2_lambda) const
        {
            return (l1_lambda + 0.5 * l2_lambda * beta) * beta;
        }

        // penalty for one group
        // l1_lambda * penalty(l2_norm(beta_j)) +
        //     l2_lambda * ridge(l2_norm(beta_j))
        inline virtual double group_penalty(const arma::rowvec& beta,
                                            const double l1_lambda,
                                            const double l2_lambda) const
        {
            const double l2_beta { l2_norm(beta) };
            return penalty0(l2_beta, l1_lambda, l2_lambda);
        }

        // penalty sum (assume all other hyper parameters are fixed) of
        // l1_lambda * penalty_factor_j * penalty(l2_norm(beta_j)) +
        //     l2_lambda * ridge(l2_norm(beta_j))
        inline virtual double regularization(const arma::mat& beta,
                                             const double l1_lambda,
                                             const double l2_lambda) const
        {
            double out { 0.0 };
            for (size_t g {0}; g < control_.penalty_factor_.n_elem; ++g) {
                out += group_penalty(beta.row(g + inter_),
                                     l1_lambda * control_.penalty_factor_(g),
                                     l2_lambda);
            }
            return out;
        }

        // objective function with regularization
        inline double objective(
            const arma::vec& inner,
            const arma::mat& beta,
            const double l1_lambda,
            const double l2_lambda
            ) const
        {
            return objective0(inner) / dn_obs_ +
                regularization(beta, l1_lambda, l2_lambda);
        }

        // optional strong rule (default: do nothing)
        inline virtual double strong_rule_rhs(const double next_lambda,
                                              const double last_lambda) const
        {
            if (false) {
                return 2 * next_lambda - last_lambda;
            }
            return 0;
        }

        // optional set gamma for scad and mcp
        inline virtual void set_gamma(const double kappa_ratio)
        {
            if (false) {
                control_.kappa_ratio_ = kappa_ratio;
            }
        }

        // core gmd step to update beta_(g)
        inline virtual void update_beta_g(arma::mat::row_iterator beta_g_it,
                                          const arma::rowvec& u_g,
                                          const double l1_lambda_g,
                                          const double l2_lambda,
                                          const double m_g) = 0;

        // run one cycle of coordinate descent over a given active set
        inline void run_one_active_cycle(
            arma::mat& beta,
            arma::vec& inner,
            arma::uvec& is_active,
            const double l1_lambda,
            const double l2_lambda,
            const bool update_active,
            const unsigned int verbose
            );

        // run complete cycles of GMD for a given active set and given lambda's
        inline void run_gmd_active_cycles(
            arma::mat& beta,
            arma::vec& inner,
            arma::uvec& is_active,
            const double l1_lambda,
            const double l2_lambda,
            const bool varying_active_set,
            const unsigned int max_iter,
            const double epsilon,
            const unsigned int verbose
            );

        // run one full cycle of coordinate descent
        inline void run_one_full_cycle(
            arma::mat& beta,
            arma::vec& inner,
            const double l1_lambda,
            const double l2_lambda,
            const unsigned int verbose
            );

        // run full cycles of GMD for given lambda's
        inline void run_gmd_full_cycles(
            arma::mat& beta,
            arma::vec& inner,
            const double l1_lambda,
            const double l2_lambda,
            const unsigned int max_iter,
            const double epsilon,
            const unsigned int verbose
            );

    public:
        // inherit constructors
        using Abclass<T_loss, T_x>::Abclass;

        // data members
        using Abclass<T_loss, T_x>::control_;
        using Abclass<T_loss, T_x>::n_obs_;
        using Abclass<T_loss, T_x>::p0_;
        using Abclass<T_loss, T_x>::p1_;
        using Abclass<T_loss, T_x>::x_;
        using Abclass<T_loss, T_x>::y_;
        using Abclass<T_loss, T_x>::ex_vertex_;
        using Abclass<T_loss, T_x>::et_npermuted_;
        using Abclass<T_loss, T_x>::coef_;
        using Abclass<T_loss, T_x>::loss_;
        using Abclass<T_loss, T_x>::penalty_;
        using Abclass<T_loss, T_x>::objective_;

        // regularization
        // the "big" enough lambda => zero coef
        double l1_lambda_max_;
        double lambda_max_;
        // did user specified a customized lambda sequence?
        bool custom_lambda_ = false;

        // function members
        using Abclass<T_loss, T_x>::rescale_coef;
        using Abclass<T_loss, T_x>::set_penalty_factor;

        // for a sequence of lambda's
        inline void fit();

    };

    // run one GMD cycle over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassGroup<T_loss, T_x>::run_one_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::uvec& is_active,
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
            const double mj { mm_lowerbound_(j) };
            const arma::rowvec old_beta_j { beta.row(j1) };
            const arma::rowvec uj { - mm_gradient(inner, j) };
            const double l1_lambda_j { l1_lambda * control_.penalty_factor_(j) };
            arma::mat::row_iterator beta_g_it { beta.begin_row(j1) };
            update_beta_g(beta_g_it, uj,
                          l1_lambda_j, l2_lambda, mj);
            arma::rowvec delta_beta_j { beta.row(j1) - old_beta_j };
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
    }

    // run CMD cycles over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassGroup<T_loss, T_x>::run_gmd_active_cycles(
        arma::mat& beta,
        arma::vec& inner,
        arma::uvec& is_active,
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
        double loss0 { objective0(inner) };
        double reg0 { regularization(beta, l1_lambda, l2_lambda) };
        double obj0 { loss0 / dn_obs_ + reg0 }, obj1 { obj0 };
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
                    Rcpp::checkUserInterrupt();
                    num_iter = ii + 1;
                    run_one_active_cycle(beta, inner, is_active_varying,
                                         l1_lambda, l2_lambda, true, verbose);
                    // if (rel_diff(beta0, beta) < epsilon) {
                    //     break;
                    // }
                    // beta0 = beta;
                    double loss1 { objective0(inner) };
                    double reg1 { regularization(beta, l1_lambda, l2_lambda) };
                    obj1 = loss1 / dn_obs_ + reg1;
                    if (verbose > 1) {
                        Rcpp::Rcout << "The objective function changed\n";
                        Rprintf("  from %7.7f (obj. %7.7f + reg. %7.7f)\n",
                                obj0, loss0, reg0);
                        Rprintf("    to %7.7f (obj. %7.7f + reg. %7.7f)\n",
                                obj1, loss1, reg1);
                        if (obj1 > obj0) {
                            Rcpp::Rcout << "Warning: "
                                        << "the objective function "
                                        << "somehow increased.\n";
                        }
                    }
                    if (std::abs(obj1 - obj0) < epsilon) {
                        break;
                    }
                    obj0 = obj1;
                    ++ii;
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
                                    << num_iter
                                    << " iteration(s)\n";
                    }
                    is_active_varying = is_active;
                    // recover the active set
                    is_active = is_active_strong;
                    ++i;
                } else {
                    break;
                }
            }
        } else {
            // regular coordinate descent
            while (i < max_iter) {
                Rcpp::checkUserInterrupt();
                ++num_iter;
                run_one_active_cycle(beta, inner, is_active,
                                     l1_lambda, l2_lambda, false, verbose);
                // if (rel_diff(beta0, beta) < epsilon) {
                //     break;
                // }
                // beta0 = beta;
                double loss1 { objective0(inner) };
                double reg1 { regularization(beta, l1_lambda, l2_lambda) };
                obj1 = loss1 / dn_obs_ + reg1;
                if (verbose > 1) {
                    Rcpp::Rcout << "The objective function changed\n";
                    Rprintf("  from %7.7f (obj. %7.7f + reg. %7.7f)\n",
                            obj0, loss0, reg0);
                    Rprintf("    to %7.7f (obj. %7.7f + reg. %7.7f)\n",
                            obj1, loss1, reg1);
                    if (obj1 > obj0) {
                        Rcpp::Rcout << "Warning: "
                                    << "the objective function "
                                    << "somehow increased.\n";
                    }
                }
                if (std::abs(obj1 - obj0) < epsilon) {
                    break;
                }
                obj0 = obj1;
                ++i;
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

    // run one full GMD cycle
    template <typename T_loss, typename T_x>
    inline void AbclassGroup<T_loss, T_x>::run_one_full_cycle(
        arma::mat& beta,
        arma::vec& inner,
        const double l1_lambda,
        const double l2_lambda,
        const unsigned int verbose
        )
    {
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
            const size_t j1 { j + inter_ };
            const double mj { mm_lowerbound_(j) };
            const arma::rowvec old_beta_j { beta.row(j1) };
            const arma::rowvec uj { - mm_gradient(inner, j) };
            const double l1_lambda_j { l1_lambda * control_.penalty_factor_(j) };
            arma::mat::row_iterator beta_g_it { beta.begin_row(j1) };
            update_beta_g(beta_g_it, uj,
                          l1_lambda_j, l2_lambda, mj);
            arma::rowvec delta_beta_j { beta.row(j1) - old_beta_j };
            arma::vec delta_vj { ex_vertex_ * delta_beta_j.t() };
            inner += x_.col(j) % delta_vj;
        }
    }

    // run full GMD cycles
    template <typename T_loss, typename T_x>
    inline void AbclassGroup<T_loss, T_x>::run_gmd_full_cycles(
        arma::mat& beta,
        arma::vec& inner,
        const double l1_lambda,
        const double l2_lambda,
        const unsigned int max_iter,
        const double epsilon,
        const unsigned int verbose
        )
    {
        size_t i {0}, num_iter {0};
        // arma::mat beta0 { beta };
        double loss0 { objective0(inner) };
        double reg0 { regularization(beta, l1_lambda, l2_lambda) };
        double obj0 { loss0 / dn_obs_ + reg0 }, obj1 { obj0 };

        // regular coordinate descent
        while (i < max_iter) {
            Rcpp::checkUserInterrupt();
            ++num_iter;
            run_one_full_cycle(beta, inner,
                               l1_lambda, l2_lambda, verbose);
            // if (rel_diff(beta0, beta) < epsilon) {
            //     break;
            // }
            // beta0 = beta;
            double loss1 { objective0(inner) };
            double reg1 { regularization(beta, l1_lambda, l2_lambda) };
            obj1 = loss1 / dn_obs_ + reg1;
            if (verbose > 1) {
                Rcpp::Rcout << "The objective function changed\n";
                Rprintf("  from %7.7f (obj. %7.7f + reg. %7.7f)\n",
                        obj0, loss0, reg0);
                Rprintf("    to %7.7f (obj. %7.7f + reg. %7.7f)\n",
                        obj1, loss1, reg1);
                if (obj1 > obj0) {
                    Rcpp::Rcout << "Warning: "
                                << "the objective function "
                                << "somehow increased.\n";
                }
            }
            if (std::abs(obj1 - obj0) < epsilon) {
                break;
            }
            obj0 = obj1;
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

    template <typename T_loss, typename T_x>
    inline void AbclassGroup<T_loss, T_x>::fit()
    {
        // set the CMD lowerbound
        set_mm_lowerbound();
        // set group weight from the control_
        set_penalty_factor();
        // set gamma
        set_gamma(control_.kappa_ratio_);
        // penalty for covariates with positive group weights only
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
        // set up lambda sequence
        if (! control_.lambda_.empty()) {
            control_.lambda_ = arma::reverse(arma::unique(control_.lambda_));
            control_.nlambda_ = control_.lambda_.n_elem;
            custom_lambda_ = true;
        }
        // if alpha = 0 and customized lambda, no need to determine lambda_max_
        if (is_ridge_only && custom_lambda_) {
            l1_lambda_max_ = - 1.0; // not well defined
            lambda_max_ = - 1.0;    // not well defined
        } else {
            // need to determine lambda_max
            one_grad_beta = gradient(one_inner);
            // get large enough lambda for zero coefs in penalty_group
            l1_lambda_max_ = 0.0;
            lambda_max_ = 0.0;
            for (arma::uvec::iterator it { penalty_group.begin() };
                 it != penalty_group.end(); ++it) {
                double tmp { l2_norm(one_grad_beta.row(*it)) };
                tmp /= control_.penalty_factor_(*it);
                if (l1_lambda_max_ < tmp) {
                    l1_lambda_max_ = tmp;
                }
            }
            lambda_max_ = l1_lambda_max_ / std::max(control_.alpha_, 1e-2);
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
        arma::uvec is_active_strong { arma::ones<arma::uvec>(p0_) };
        // 1) no need to consider possible constant covariates
        is_active_strong.elem(arma::find(mm_lowerbound_ <= 0.0)).zeros();
        // 2) only need to estimate beta not in the penalty group
        is_active_strong.elem(penalty_group).zeros();
        if (control_.intercept_) {
            run_gmd_active_cycles(one_beta,
                                  one_inner,
                                  is_active_strong,
                                  l1_lambda,
                                  l2_lambda,
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
                run_gmd_full_cycles(one_beta,
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
        // exclude constant covariates from penalty group
        // so that they will not be considered as active by strong rule at all
        penalty_group = penalty_group.elem(arma::find(mm_lowerbound_ > 0.0));
        // for strong rule
        double one_strong_rhs { 0.0 }, old_lambda { l1_lambda_max_ };
        // main loop: for each lambda
        for (size_t li { 0 }; li < control_.lambda_.n_elem; ++li) {
            double lambda_li { control_.lambda_(li) };
            l1_lambda = control_.alpha_ * lambda_li;
            l2_lambda = (1 - control_.alpha_) * lambda_li;
            // early exit for lambda greater than lambda_max_
            // note that lambda is sorted
            if (l1_lambda >= l1_lambda_max_) {
                coef_.slice(li) = rescale_coef(one_beta);
                loss_(li) = objective0(one_inner);
                penalty_(li) = regularization(one_beta, l1_lambda, l2_lambda);
                objective_(li) = loss_(li) / dn_obs_ + penalty_(li);
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
                one_strong_rhs = control_.penalty_factor_(*it) *
                    strong_rule_rhs(l1_lambda, old_lambda);
                if (one_strong_lhs >= one_strong_rhs) {
                    is_active_strong(*it) = 1;
                }
            }
            old_lambda = l1_lambda; // for next iteration
            bool kkt_failed { true };
            // eventually, strong rule will guess correctly
            while (kkt_failed) {
                arma::uvec is_active_strong_old { is_active_strong };
                arma::uvec is_strong_rule_failed {
                    arma::zeros<arma::uvec>(is_active_strong.n_elem)
                };
                // update beta
                run_gmd_active_cycles(one_beta,
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
                // cache some variables inside of mm_gradient
                arma::vec tmp_vec;
                if (penalty_group.n_elem > 0) {
                    const arma::vec inner_grad = loss_derivative(one_inner);
                    tmp_vec = control_.obs_weight_ % inner_grad;
                }
                for (arma::uvec::iterator it { penalty_group.begin() };
                     it != penalty_group.end(); ++it) {
                    if (is_active_strong_old(*it) > 0) {
                        continue;
                    }
                    arma::vec tmp_vec_it { x_.col(*it) % tmp_vec };
                    arma::rowvec tmp_mm_grad { tmp_vec_it.t() * ex_vertex_ };
                    double tmp_l2 { l2_norm(tmp_mm_grad) / dn_obs_ };
                    if (tmp_l2 > l1_lambda * control_.penalty_factor_(*it)) {
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
            // check if any permuted predictors are selected
            if (et_npermuted_ > 0) {
                if (control_.verbose_ > 0) {
                    msg("[ET] check if any pseudo-predictors was selected.");
                }
                // assume the last (permuted) predictors are inactive
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
                        objective_ = objective_(0);
                    } else {
                        // discard the estimates from this itaration
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

}

#endif /* ABCLASS_ABCLASS_GROUP_H */
