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

#ifndef ABCLASS_ABCLASS_CD_H
#define ABCLASS_ABCLASS_CD_H

#include <RcppArmadillo.h>

#include "AbclassLinear.h"
#include "MarginLoss.h"
#include "utils.h"

namespace abclass
{
    // angle-based classifier with coordinate-descent type of algorithms
    template <typename T_loss, typename T_x>
    class AbclassCD : public AbclassLinear<T_loss, T_x>
    {
    protected:
        // data members
        using AbclassLinear<T_loss, T_x>::mm_lowerbound0_;
        using AbclassLinear<T_loss, T_x>::mm_lowerbound_;
        using AbclassLinear<T_loss, T_x>::null_loss_;
        using AbclassLinear<T_loss, T_x>::last_loss_;
        using AbclassLinear<T_loss, T_x>::last_penalty_;
        using AbclassLinear<T_loss, T_x>::last_obj_;
        using AbclassLinear<T_loss, T_x>::last_eps_;

        // function members
        using AbclassLinear<T_loss, T_x>::dloss_dbeta;
        using AbclassLinear<T_loss, T_x>::iter_dloss_dbeta;
        using AbclassLinear<T_loss, T_x>::iter_dloss_df;
        using AbclassLinear<T_loss, T_x>::iter_loss;
        using AbclassLinear<T_loss, T_x>::rescale_coef;

        // cache
        size_t active_ncol_;
        bool converged_;

        // specifying if a blockwise CD should be used
        inline virtual void set_active_ncol()
        {
            active_ncol_ = static_cast<size_t>(data_.km1_);
        }

        // penalty function for theta >= 0 (default: lasso)
        inline virtual double penalty0(const double theta,
                                       const double l1_lambda,
                                       const double l2_lambda) const
        {
            return (l1_lambda + 0.5 * l2_lambda * theta) * theta;
        }

        // penalty function for the coefficient vector of one covariate
        // e.g., elastic-net =
        //   l1_lambda * l1_norm(beta_j) +
        //     0.5 * l2_lambda * l2_norm^2(beta_j),
        //   where
        //     l1_lambda = alpha * lambda (* optional penalty factor),
        //     l2_lambda = (1-alpha) * lambda
        inline virtual double penalty1(const arma::rowvec& beta,
                                       const double l1_lambda,
                                       const double l2_lambda) const
        {
            double out { 0.0 };
            for (size_t k {0}; k < beta.n_elem; ++k) {
                out += penalty0(std::abs(beta(k)),
                                l1_lambda, l2_lambda);
            }
            return out;
        }

        // regularization for the coefficient matrix
        inline virtual double regularization(const arma::mat& beta,
                                             const double l1_lambda,
                                             const double l2_lambda) const
        {
            double out { 0.0 };
            for (size_t g {0}; g < control_.penalty_factor_.n_elem; ++g) {
                out += penalty1(beta.row(g + data_.inter_),
                                l1_lambda * control_.penalty_factor_(g),
                                l2_lambda);
            }
            return out;
        }

        // objective = iter_loss / n + regularization
        // inline virtual double objective(const arma::mat& beta,
        //                                 const double l1_lambda,
        //                                 const double l2_lambda) const
        // {
        //     return iter_loss() * data_.div_n_obs_ +
        //         regularization(beta, l1_lambda, l2_lambda);
        // }

        inline arma::vec gen_penalty_factor(
            const arma::vec& penalty_factor = arma::vec()
            ) const
        {
            if (penalty_factor.is_empty()) {
                return arma::ones(data_.p0_);
            }
            if (penalty_factor.n_elem == data_.p0_) {
                if (arma::any(penalty_factor < 0.0)) {
                    throw std::range_error(
                        "The 'penalty_factor' cannot be negative.");
                }
                return penalty_factor;
            }
            // else
            throw std::range_error("Incorrect length of the 'penalty_factor'.");
        }

        // for intercept
        inline arma::rowvec mm_gradient0() const
        {
            const arma::mat grad { iter_dloss_df() };
            return arma::mean(grad, 0);
        }

        inline double mm_gradient0(const unsigned int k) const
        {
            const arma::vec grad_k { iter_dloss_df(k) };
            return arma::accu(grad_k) * data_.div_n_obs_;
        }

        // define gradient function at (g, k) for the given inner product
        inline double mm_gradient(const unsigned int g,
                                  const unsigned int k)
        {
            const arma::vec grad_k { iter_dloss_dbeta(g, k) };
            return arma::accu(grad_k) * data_.div_n_obs_;
        }

        // define gradient function for g-th predictor
        inline arma::rowvec mm_gradient(const unsigned int g)
        {
            const arma::mat grad { iter_dloss_dbeta(g) };
            return arma::mean(grad, 0);
        }

        // gradient matrix regarding coef only (excluding intercept)
        inline arma::mat gradient() const
        {
            arma::mat out(data_.p0_, data_.km1_);
            arma::mat grad { iter_dloss_df() };
            for (size_t k {0}; k < data_.km1_; ++k) {
                for (size_t j {0}; j < data_.p0_; ++j) {
                    out(j, k) = arma::dot(grad.col(k), data_.x_.col(j)) *
                        data_.div_n_obs_;
                }
            }
            return out;
        }

        // determine the large-enough l1 lambda that results in zero coef's
        inline virtual void set_lambda_max(const arma::uvec& positive_penalty)
        {
            arma::mat one_grad_beta { arma::abs(gradient()) };
            // get large enough lambda for zero coefs in positive_penalty
            l1_lambda_max_ = 0.0;
            for (arma::uvec::const_iterator it { positive_penalty.begin() };
                 it != positive_penalty.end(); ++it) {
                double tmp { one_grad_beta.row(*it).max() };
                tmp /= control_.penalty_factor_(*it);
                if (l1_lambda_max_ < tmp) {
                    l1_lambda_max_ = tmp;
                }
            }
            lambda_max_ =  l1_lambda_max_ /
                std::max(control_.ridge_alpha_, control_.lambda_max_alpha_min_);
        }

        // optional set gamma for non-convex penalty (e.g., scad and mcp)
        inline virtual void set_gamma(const double kappa)
        {
            control_.ncv_kappa_ = kappa;
            // should be more for actual penalty
        }

        // optional strong rule
        inline virtual double strong_rule_lhs(const double beta_gk) const
        {
            return std::abs(beta_gk);
        }
        inline virtual double strong_rule_lhs(const arma::rowvec& beta_g) const
        {
            return l2_norm(beta_g);
        }

        inline virtual double strong_rule_rhs(const double next_lambda,
                                              const double last_lambda) const
        {
            // default to do nothing
            if (false) {
                return 2 * next_lambda - last_lambda;
            }
            return 0.0;
        }

        inline virtual void apply_strong_rule(
            arma::umat& is_active_strong,
            const double next_lambda,
            const double last_lambda,
            const arma::uvec positive_penalty)
        {
            // update active set by strong rule
            arma::mat one_grad_beta { gradient() };
            double one_strong_rhs { strong_rule_rhs(next_lambda, last_lambda) };
            for (size_t j { 0 }; j < active_ncol_; ++j) {
                for (arma::uvec::const_iterator it { positive_penalty.begin() };
                     it != positive_penalty.end(); ++it) {
                    if (is_active_strong(*it, j) > 0) {
                        continue;
                    }
                    double sr_lhs { strong_rule_lhs(one_grad_beta(*it, j)) };
                    if (sr_lhs >= control_.penalty_factor_(*it) *
                        one_strong_rhs) {
                        is_active_strong(*it, j) = 1;
                    }
                }
            }
        }

        // kkt condition
        inline virtual arma::umat is_kkt_failed(
            const arma::umat& is_active_strong,
            const arma::uvec& positive_penalty,
            const double l1_lambda) const
        {
            arma::umat is_strong_rule_failed(arma::size(is_active_strong),
                                             arma::fill::zeros);
            arma::mat dloss_df;
            if (positive_penalty.n_elem > 0) {
                dloss_df = iter_dloss_df();
            }
            for (arma::uvec::const_iterator it { positive_penalty.begin() };
                 it != positive_penalty.end(); ++it) {
                for (size_t j { 0 }; j < active_ncol_; ++j) {
                    if (is_active_strong(*it, j) > 0) {
                        continue;
                    }
                    const arma::vec x_g { data_.x_.col(*it) };
                    const arma::vec dj { dloss_df.col(j) };
                    const arma::vec dloss_dbeta_ { dloss_dbeta(dj, x_g) };
                    const double tmp {
                        arma::accu(dloss_dbeta_) * data_.div_n_obs_
                    };
                    if (std::abs(tmp) > l1_lambda *
                        control_.penalty_factor_(*it)) {
                        // update active set
                        is_strong_rule_failed(*it, j) = 1;
                    }
                }
            }
            return is_strong_rule_failed;
        }

        // default: individual update step for beta
        inline virtual void update_beta_gk(arma::mat& beta,
                                           const size_t k,
                                           const size_t g,
                                           const size_t g1,
                                           const double l1_lambda,
                                           const double l2_lambda)
        {
            const double old_beta_g1k { beta(g1, k) };
            const double d_gk { mm_gradient(g, k) };
            // if mm_lowerbound = 0 and l1_lambda > 0, numer will be 0
            const double m_g { mm_lowerbound_(g) };
            const double u_g { m_g * beta(g1, k) - d_gk };
            const double tmp {
                std::abs(u_g) - l1_lambda * control_.penalty_factor_(g)
            };
            if (tmp > 0.0) {
                const double numer { tmp * sign(u_g) };
                const double denom { m_g + l2_lambda };
                // update beta
                beta(g1, k) = std::max(
                    control_.lower_limit_(g, k),
                    std::min(control_.upper_limit_(g, k),
                             numer / denom));
            } else {
                beta(g1, k) = 0.0;
            }
            // update pred_f and inner
            const double delta_beta { beta(g1, k) - old_beta_g1k };
            if (delta_beta != 0.0) {
                if constexpr (std::is_base_of_v<MarginLoss, T_loss>) {
                    data_.iter_inner_ += delta_beta * data_.iter_vk_xg_;
                } else {
                    data_.iter_pred_f_.col(k) += delta_beta * data_.x_.col(g);
                }
                last_eps_ = std::max(last_eps_, m_g * delta_beta * delta_beta);
            }
        }

        // for 1) individual or bi-level regularization (default)
        //  or 2) group-wise regularization (needs overriding)
        // run one cycle of coordinate descent over a given active set
        inline virtual void run_one_active_cycle(
            arma::mat& beta,
            arma::umat& is_active,
            const double l1_lambda,
            const double l2_lambda,
            const bool update_active,
            const unsigned int verbose);

        // run cycles a given active set and given lambda's until convergence
        inline void run_active_cycles(arma::mat& beta,
                                      arma::umat& is_active,
                                      const double l1_lambda,
                                      const double l2_lambda,
                                      const bool varying_active_set,
                                      const unsigned int max_iter,
                                      const double epsilon,
                                      const unsigned int verbose);

    public:
        // inherit constructors
        using AbclassLinear<T_loss, T_x>::AbclassLinear;

        // specifics for template inheritance
        // from Abclass
        using AbclassLinear<T_loss, T_x>::data_;
        using AbclassLinear<T_loss, T_x>::control_;
        using AbclassLinear<T_loss, T_x>::loss_fun_;

        // from AbclassLinear
        using AbclassLinear<T_loss, T_x>::set_mm_lowerbound;
        using AbclassLinear<T_loss, T_x>::coef_;
        using AbclassLinear<T_loss, T_x>::loss_;
        using AbclassLinear<T_loss, T_x>::penalty_;
        using AbclassLinear<T_loss, T_x>::objective_;
        using AbclassLinear<T_loss, T_x>::n_iter_;

        // tuning by cross-validation
        arma::mat cv_accuracy_;
        arma::vec cv_accuracy_mean_;
        arma::vec cv_accuracy_sd_;

        // tuning by ET-Lasso
        unsigned int et_npermuted_ { 0 }; // number of permuted predictors
        arma::uvec et_vs_;                // indices of selected predictors

        // one time value for one stage
        // the smallest lambda before selection of any random predictors
        double et_l1_lambda0_; // the last lambda before the cutoff
        double et_l1_lambda1_; // the cutoff point

        // to save values from all the stages for output
        arma::vec et_l1_lambda0_vec_;
        arma::vec et_l1_lambda1_vec_;

        // regularization
        // the "big" enough lambda => zero coef unless alpha = 0
        double l1_lambda_max_;
        double lambda_max_;

        // setter for penalty factors
        inline void set_penalty_factor(
            const arma::vec& penalty_factor = arma::vec()
            )
        {
            if (penalty_factor.n_elem > 0) {
                control_.penalty_factor_ = gen_penalty_factor(penalty_factor);
            } else {
                control_.penalty_factor_ = gen_penalty_factor(
                    control_.penalty_factor_);
            }
        }

        // for a sequence of lambda's
        inline void fit();

    };

    // run one update step over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassCD<T_loss, T_x>::run_one_active_cycle(
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
                        << is_active << "\n";
        };
        // for intercept
        if (control_.intercept_) {
            arma::rowvec delta_beta0 { - mm_gradient0() / mm_lowerbound0_ };
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
                    arma::max(mm_lowerbound0_ * delta_beta0 % delta_beta0));
            }
        }
        // for predictors
        for (size_t g { 0 }; g < data_.p0_; ++g) {
            const size_t g1 { g + data_.inter_ };
            for (size_t k {0}; k < data_.km1_; ++k) {
                if (is_active(g, k) == 0) {
                    continue;
                }
                // update beta and inner
                update_beta_gk(beta, k, g, g1, l1_lambda, l2_lambda);
                // update active
                if (update_active) {
                    // check if it has been shrinkaged to zero
                    if (beta(g1, k) == 0.0) {
                        is_active(g, k) = 0;
                    }
                    // is_active(g, k) must be 1 to get here
                }
            }
        }
        if (verbose > 2) {
            Rcpp::Rcout << "\nUpdated beta after one cycle:\n"
                        << beta << "\n"
                        << "\nThe active set of updated beta:\n"
                        << is_active << "\n";
        };
        ++n_iter_;
    }

    // run CMD cycles over active sets
    template <typename T_loss, typename T_x>
    inline void AbclassCD<T_loss, T_x>::run_active_cycles(
        arma::mat& beta,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
        const bool varying_active_set,
        const unsigned int max_iter,
        const double epsilon,
        const unsigned int verbose
        )
    {
        converged_ = false;
        size_t num_iter {0};
        // use active-set if p > n ("helps when p >> n")
        if (varying_active_set) {
            arma::umat is_active_strong { is_active },
                is_active_varying { is_active };
            if (verbose > 0) {
                Rcpp::Rcout << "The size of active set from strong rule: "
                            << l1_norm(is_active_strong)
                            << "\n";
            }
            while (n_iter_ < max_iter) {
                num_iter = 0;
                // cycles over the active set
                while (n_iter_ < max_iter) {
                    Rcpp::checkUserInterrupt();
                    run_one_active_cycle(beta,
                                         is_active_varying,
                                         l1_lambda,
                                         l2_lambda,
                                         true,
                                         verbose);
                    ++num_iter;
                    // optional: throw warning if objective function increases
                    if (verbose > 1 || control_.adjust_mm_) {
                        double loss1 { iter_loss() };
                        double pen1 { regularization(beta, l1_lambda, l2_lambda) };
                        double obj1 { loss1 * data_.div_n_obs_ + pen1 };
                        if (verbose > 1) {
                            Rcpp::Rcout << "The objective function changed\n";
                            Rprintf("  from %7.7f (iter_loss: %7.7f + penalty: %7.7f)\n",
                                    last_obj_, last_loss_ * data_.div_n_obs_, last_penalty_);
                            Rprintf("    to %7.7f (iter_loss: %7.7f + penalty: %7.7f)\n",
                                    obj1, loss1 * data_.div_n_obs_, pen1);
                        }
                        if (last_obj_ < obj1) {
                            if (verbose > 1) {
                                Rcpp::Rcout << "Notice: the objective increased.\n";
                            }
                            if (control_.adjust_mm_) {
                                mm_lowerbound0_ = mm_lowerbound0_ * 2.0;
                                mm_lowerbound_ = mm_lowerbound_ * 2.0;
                            }
                        }
                        last_loss_ = loss1;
                        last_obj_ = obj1;
                        last_penalty_ = pen1;
                    }
                    if (last_eps_ < epsilon) {
                        converged_ = true;
                        break;
                    }
                }
                // run a full cycle over the converged beta
                run_one_active_cycle(beta,
                                     is_active,
                                     l1_lambda,
                                     l2_lambda,
                                     true,
                                     verbose);
                ++num_iter;
                // check two active sets coincide
                if (arma::accu(arma::any(is_active_varying != is_active))) {
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
                    // however, quit anyway if beta is converged
                    if (last_eps_ < epsilon) {
                        converged_ = true;
                        break;
                    }
                    // otherwise, recover the active set
                    is_active_varying = is_active_strong;
                    is_active = is_active_strong;
                } else {
                    break;
                }
            }
        } else {
            // regular coordinate descent
            while (n_iter_ < max_iter) {
                Rcpp::checkUserInterrupt();
                run_one_active_cycle(beta,
                                     is_active,
                                     l1_lambda,
                                     l2_lambda,
                                     false,
                                     verbose);
                ++num_iter;
                // optional: throw warning if objective function increases
                if (verbose > 1 || control_.adjust_mm_) {
                    double loss1 { iter_loss() };
                    double pen1 { regularization(beta, l1_lambda, l2_lambda) };
                    double obj1 { loss1 * data_.div_n_obs_ + pen1 };
                    if (verbose > 1) {
                        Rcpp::Rcout << "The objective function changed\n";
                        Rprintf("  from %7.7f (iter_loss: %7.7f + penalty: %7.7f)\n",
                                last_obj_, last_loss_ * data_.div_n_obs_, last_penalty_);
                        Rprintf("    to %7.7f (iter_loss: %7.7f + penalty: %7.7f)\n",
                                obj1, loss1 * data_.div_n_obs_, pen1);
                    }
                    if (last_obj_ < obj1) {
                        if (verbose > 1) {
                            Rcpp::Rcout << "Warning: "
                                        << "the function objective "
                                        << "somehow increased.\n";
                        }
                        if (control_.adjust_mm_) {
                            mm_lowerbound0_ = mm_lowerbound0_ * 2.0;
                            mm_lowerbound_ = mm_lowerbound_ * 2.0;
                        }
                    }
                    last_loss_ = loss1;
                    last_obj_ = obj1;
                    last_penalty_ = pen1;
                }
                if (last_eps_ < epsilon) {
                    converged_ = true;
                    break;
                }
            }
        }
        if (verbose > 0) {
            if (converged_) {
                Rcpp::Rcout << "Outer loop converged over the active set after "
                            << num_iter
                            << " iteration(s);\n";
                Rcpp::Rcout << "The size of active set after convergence is "
                            << l1_norm(is_active) << ".\n\n";
            } else {
                msg("Outer loop reached the maximum number of iteratons.");
            }
        }
    }

    // for a sequence of lambda's
    // lambda * (alpha * lasso + (1 - alpha) / 2 * ridge)
    template <typename T_loss, typename T_x>
    inline void AbclassCD<T_loss, T_x>::fit()
    {
        // set the CMD lowerbound
        set_mm_lowerbound();
        // set penalty factor from the control_
        set_penalty_factor(control_.penalty_factor_);
        // set gamma
        set_gamma(control_.ncv_kappa_);
        // set cache to help determine update steps
        set_active_ncol();
        // initialize total number of iterations
        n_iter_ = 0;
        // penalty for covariates with positive penalty factors only
        arma::uvec positive_penalty {
            arma::find(control_.penalty_factor_ > 0.0)
        };
        // initialize
        if constexpr (std::is_base_of_v<MarginLoss, T_loss>) {
            if (control_.has_offset_) {
                data_.iter_inner_ = arma::sum(
                    data_.ex_vertex_ % control_.offset_, 1);
            } else {
                data_.iter_inner_ = arma::zeros(data_.n_obs_);
            }
        } else {
            if (control_.has_offset_) {
                data_.iter_pred_f_ = control_.offset_;
            } else {
                data_.iter_pred_f_ = arma::zeros(data_.n_obs_, data_.km1_);
            }
        }
        arma::mat one_beta(data_.p1_, data_.km1_);
        const bool is_ridge_only {
            isAlmostEqual(control_.ridge_alpha_, 0.0)
        };
        double l1_lambda { 0.0 }, l2_lambda { 0.0 };
        // if alpha = 0 and customized lambda
        if (is_ridge_only && control_.custom_lambda_) {
            l1_lambda_max_ = - 1.0; // not well defined
            lambda_max_ = - 1.0;    // not well defined
        } else {
            // need to determine lambda_max
            set_lambda_max(positive_penalty);
            // set up lambda sequence
            if (! control_.custom_lambda_) {
                double log_lambda_max { std::log(lambda_max_) };
                double log_lambda_min { 0.0 };
                if (control_.lambda_min_ > 0.0) {
                    log_lambda_min = std::log(control_.lambda_min_);
                } else {
                    log_lambda_min = log_lambda_max +
                        std::log(control_.lambda_min_ratio_);
                }
                control_.lambda_ = arma::exp(
                    arma::linspace(log_lambda_max, log_lambda_min,
                                   control_.nlambda_));
            }
        }
        // initialize the estimate cube
        coef_ = arma::cube(data_.p1_, data_.km1_, control_.lambda_.n_elem,
                           arma::fill::zeros);
        objective_ = penalty_ = loss_ = arma::zeros(control_.lambda_.n_elem);
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::umat is_active_strong(data_.p0_, active_ncol_, arma::fill::ones);
        // 1) no need to consider possible constant covariates
        is_active_strong.rows(data_.x_skip_).zeros();
        arma::umat is_active_strong2; // pure ridge
        if (is_ridge_only) {
            is_active_strong2 = is_active_strong;
        }
        // 2) only need to estimate beta not in the penalty group
        is_active_strong.rows(positive_penalty).zeros();
        // set up epsilon0
        last_loss_ = null_loss_ = iter_loss();
        last_penalty_ = 0.0;
        last_obj_ = last_loss_ * data_.div_n_obs_;
        double epsilon0 {
            exp_log_sum(control_.epsilon_, null_loss_)
        };
        if (control_.intercept_) {
            // only need to estimate intercept
            run_active_cycles(one_beta,
                              is_active_strong,
                              0, // does not matter
                              0, // does not matter
                              false,
                              epsilon0,
                              control_.epsilon_,
                              control_.verbose_);
            last_loss_ = null_loss_ = iter_loss();
            epsilon0 = exp_log_sum(control_.epsilon_, null_loss_);
            last_obj_ = last_loss_ * data_.div_n_obs_;
        }
        // for pure ridge penalty
        if (is_ridge_only) {
            for (size_t li { 0 }; li < control_.lambda_.n_elem; ++li) {
                l2_lambda = control_.lambda_(li);
                run_active_cycles(one_beta,
                                  is_active_strong2,
                                  l1_lambda,
                                  l2_lambda,
                                  false,
                                  control_.max_iter_,
                                  epsilon0,
                                  control_.verbose_);
                coef_.slice(li) = rescale_coef(one_beta);
                loss_(li) = iter_loss();
                penalty_(li) = regularization(one_beta, l1_lambda, l2_lambda);
                objective_(li) = loss_(li) * data_.div_n_obs_ + penalty_(li);
                // if max iter is reached
                if (! converged_) {
                    size_t li1 { li + 1 };
                    coef_ = coef_.head_slices(li1);
                    control_.lambda_ = control_.lambda_.head(li1);
                    loss_ = loss_.head(li1);
                    penalty_ = penalty_.head(li1);
                    objective_ = objective_.head(li1);
                    break;
                }
            }
            data_.reset_cache();
            return;             // early exit
        }
        // else, not just ridge penalty with l1_lambda > 0
        // exclude constant covariates from penalty group
        // so that they will not be considered as active by strong rule at all
        positive_penalty = positive_penalty.elem(
            arma::find(mm_lowerbound_ > 0.0));
        // for strong rule
        double old_l1_lambda { l1_lambda_max_ };
        // main loop: for each lambda
        for (size_t li { 0 }; li < control_.lambda_.n_elem; ++li) {
            double lambda_li { control_.lambda_(li) };
            l1_lambda = lambda_li * control_.ridge_alpha_;
            l2_lambda = lambda_li * (1 - control_.ridge_alpha_);
            // early exit for lambda greater than lambda_max_
            // note that lambda is sorted
            if (l1_lambda >= l1_lambda_max_) {
                coef_.slice(li) = rescale_coef(one_beta);
                loss_(li) = null_loss_;
                penalty_(li) = regularization(one_beta, l1_lambda, l2_lambda);
                objective_(li) = loss_(li) * data_.div_n_obs_ + penalty_(li);
                continue;
            }
            // update active set by strong rule
            apply_strong_rule(is_active_strong, l1_lambda,
                              old_l1_lambda, positive_penalty);
            bool kkt_failed { true };
            // eventually, strong rule will guess correctly
            while (kkt_failed) {
                arma::umat is_active_strong_old { is_active_strong };
                // update beta
                run_active_cycles(one_beta,
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
                arma::umat is_strong_rule_failed {
                    is_kkt_failed(is_active_strong_old,
                                  positive_penalty,
                                  l1_lambda)
                };
                if (arma::accu(is_strong_rule_failed) > 0) {
                    is_active_strong = is_active_strong_old ||
                        is_strong_rule_failed;
                    if (control_.verbose_ > 0) {
                        Rcpp::Rcout << "The strong rule failed.\n"
                                    << "Expended the active set;"
                                    << "\n  The size of old active set: "
                                    << l1_norm(is_active_strong_old)
                                    << "\n  The size of new active set: "
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
                if (! permuted_beta.is_zero()) {
                    if (li == 0) {
                        msg("Warning: Failed to tune by ET-lasso; ",
                            "selected pseudo-predictor(s) by ",
                            "the largest lamabda specified; ",
                            "the returned solution may not be sensible.\n",
                            "Suggestion: increase 'lambda', ",
                            "'lambda_min_ratio' or 'nlambda'?\n");
                        coef_ = coef_.head_slices(1);
                        loss_ = loss_(0);
                        penalty_ = penalty_(0);
                        objective_(0) = loss_(0) * data_.div_n_obs_ + penalty_(0);
                    } else {
                        // discard the estimates from this itaration
                        coef_ = coef_.head_slices(li);
                        loss_ = loss_.head(li);
                        penalty_ = penalty_.head(li);
                        objective_ = objective_.head(li);
                    }
                    et_l1_lambda0_ = old_l1_lambda;
                    et_l1_lambda1_ = l1_lambda;
                    if (control_.verbose_ > 0) {
                        msg("[ET] selected pseudo-predictor(s).\n");
                    }
                    break;
                }
                if (control_.verbose_ > 0) {
                    msg("[ET] none of pseudo-predictors was selected.\n");
                }
                if (li == control_.lambda_.n_elem - 1) {
                    msg("Warning: Failed to tune by ET-lasso; ",
                        "no pseudo-predictors selected ",
                        "by the smallest lambda.\n",
                        "Suggestion: decrease 'lambda' or 'lambda_min_ratio'?");
                    et_l1_lambda0_ = l1_lambda;
                    et_l1_lambda1_ = - 1.0; // tell fit() to ignore
                }
            }
            coef_.slice(li) = rescale_coef(one_beta);
            loss_(li) = iter_loss();
            penalty_(li) = regularization(one_beta, l1_lambda, l2_lambda);
            objective_(li) = loss_(li) * data_.div_n_obs_ + penalty_(li);
            // if max iter is reached
            if (! converged_) {
                size_t li1 { li + 1 };
                coef_ = coef_.head_slices(li1);
                control_.lambda_ = control_.lambda_.head(li1);
                loss_ = loss_.head(li1);
                penalty_ = penalty_.head(li1);
                objective_ = objective_.head(li1);
                break;
            }
            old_l1_lambda = l1_lambda; // for next iteration
        }
        data_.reset_cache();
    }

}  // abclass

#endif /* ABCLASS_ABCLASS_CD_H */
