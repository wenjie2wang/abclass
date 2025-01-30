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

#ifndef ABCLASS_CONTROL_H
#define ABCLASS_CONTROL_H

#include <RcppArmadillo.h>
#include <stdexcept>
#include "utils.h"

namespace abclass
{
    // all the control paramters
    class Control
    {
    public:
        // model
        arma::vec obs_weight_ { arma::vec() }; // observational weights
        bool custom_obs_weight_ { false };     // is obs_weight_ customized
        bool intercept_ { true };              // if to contrain intercepts

        // offset
        bool has_offset_ { false };        // to avoid some computation
        arma::mat offset_ { arma::mat() }; // for the decision functions

        // lower/upper limits for coefficients in linear learning
        arma::mat lower_limit_ { arma::mat() };
        arma::mat upper_limit_ { arma::mat() };

        // reward for the outcome-weighted learning
        arma::vec owl_reward_ { arma::vec() };

        // regularization
        //   true if user specified a customized lambda sequence
        bool custom_lambda_ { false };
        arma::vec lambda_  { arma::vec() };
        unsigned int nlambda_ { 20 };
        double lambda_min_ratio_ { 0.01 };
        double lambda_min_ { - 1.0 };
        double lambda_max_alpha_min_ { 0.01 };
        //   adaptive penalty factor for each covariate
        arma::vec penalty_factor_ { arma::vec() };
        //   ridge
        double ridge_alpha_ { 1.0 };
        //   scad, mcp
        double ncv_kappa_ { 0.1 };   // parameter to set gamma
        double ncv_gamma_ { - 1.0 }; // gamma for group non-convex penalty
        //   group exponential penalty
        double gel_tau_ { 0.33 };
        //   mellowmax penalty
        double mellowmax_omega_ { 1 };

        // tuning
        //   cross-validation
        unsigned int cv_nfolds_ { 0 };
        bool cv_stratified_ { true };
        arma::uvec cv_strata_ { arma::uvec() };
        unsigned int cv_alignment_ { 0 };
        //   ET-lasso
        unsigned int et_nstages_ { 0 };

        // optimization
        unsigned int max_iter_ { 10000 };  // maximum number of iterations
        double epsilon_ { 1e-7 };          // tolerance to check convergence
        bool varying_active_set_ { true }; // if active set should be adaptive
        // monitor convergecen and adjust mm lowerbound if needed
        bool adjust_mm_ { false };
        bool standardize_ { true };        // is x_ standardized (column-wise)
        unsigned int verbose_ { 0 };

        // for ranking
        bool query_weight_ { false };
        bool delta_weight_ { false };
        bool delta_adaptive_ { false };
        unsigned int delta_max_iter_ { 10 };

        // default constructor
        Control() {}

        Control(const unsigned int max_iter,
                const double epsilon,
                const bool standardize = true,
                const unsigned int verbose = 0)
        {
            if (is_lt(epsilon, 0.0)) {
                throw std::range_error("The 'epsilon' cannot be negative.");
            }
            max_iter_ = max_iter;
            epsilon_ = epsilon;
            standardize_ = standardize;
            verbose_ = verbose;
        }

        // individual setters
        inline Control* set_intercept(const bool intercept)
        {
            intercept_ = intercept;
            return this;
        }
        inline Control* set_weight(const arma::vec& obs_weight)
        {
            obs_weight_ = obs_weight;
            return this;
        }
        template <typename T=arma::mat>
        inline Control* set_offset(const T& offset)
        {
            offset_ = arma::mat(offset);
            if (offset_.n_elem > 0) {
                has_offset_ = true;
            }
            return this;
        }
        template <typename T=arma::mat>
        inline Control* set_lower_limit(const T& lower_limit)
        {
            lower_limit_ = arma::mat(lower_limit);
            return this;
        }
        template <typename T=arma::mat>
        inline Control* set_upper_limit(const T& upper_limit)
        {
            upper_limit_ = arma::mat(upper_limit);
            return this;
        }
        inline Control* set_standardize(const bool standardize)
        {
            standardize_ = standardize;
            return this;
        }
        inline Control* set_verbose(const unsigned int verbose)
        {
            verbose_ = verbose;
            return this;
        }
        // for outcome-weighted learning
        inline Control* set_owl_reward(const arma::vec& reward)
        {
            if (reward.n_elem == 0 || reward.is_zero()) {
                owl_reward_ = arma::vec();
                return this;
            }
            owl_reward_ = reward;
            return this;
        }

        // regularization
        inline Control* reg_path(const unsigned int nlambda,
                                 const double lambda_min_ratio,
                                 const arma::vec& penalty_factor = arma::vec(),
                                 const bool varying_active_set = true,
                                 const bool adjust_mm = false)
        {
            if (is_le(lambda_min_ratio, 0.0)) {
                throw std::range_error(
                    "The 'lambda_min_ratio' must be positive.");
            }
            lambda_min_ratio_ = lambda_min_ratio;
            nlambda_ = nlambda;
            penalty_factor_ = penalty_factor;
            varying_active_set_ = varying_active_set;
            adjust_mm_ = adjust_mm;
            return this;
        }
        inline Control* reg_lambda(const arma::vec& lambda = arma::vec())
        {
            lambda_ = lambda;
            if (! lambda_.empty()) {
                custom_lambda_ = true;
                lambda_ = arma::reverse(arma::unique(lambda));
                nlambda_ = lambda_.n_elem;
            } else {
                custom_lambda_ = false;
                lambda_.clear();
            }
            return this;
        }
        inline Control* reg_lambda_min(const double lambda_min = - 1.0)
        {
            lambda_min_ = lambda_min;
            return this;
        }
        inline Control* reg_ridge(const double alpha,
                                  const double lambda_max_alpha_min = 0.01)
        {
            // check alpha
            if ((alpha < 0.0) || (alpha > 1.0)) {
                throw std::range_error("The 'alpha' must be between 0 and 1.");
            }
            if (lambda_max_alpha_min <= 0.0) {
                throw std::range_error(
                    "The 'lambda_max_alpha_min' must be positive.");
            }
            ridge_alpha_ = alpha;
            lambda_max_alpha_min_ = lambda_max_alpha_min;
            return this;
        }
        inline Control* reg_ncv(const double kappa = 0.9)
        {
            // kappa must be in (0, 1)
            if (is_le(kappa, 0.0) || is_ge(kappa, 1.0)) {
                throw std::range_error("The 'kappa' must be in (0, 1).");
            }
            ncv_kappa_ = kappa;
            return this;
        }
        inline Control* reg_gel(const double tau = 0.33)
        {
            if (is_le(tau, 0.0)) {
                throw std::range_error("The 'tau' must be positive.");
            }
            gel_tau_ = tau;
            return this;
        }
        inline Control* reg_mellowmax(const double omega = 10.0)
        {
            if (is_le(omega, 0.0)) {
                throw std::range_error("The 'omega' must be positive.");
            }
            mellowmax_omega_ = omega;
            return this;
        }

        // tuning
        inline Control* tune_cv(const unsigned int nfolds,
                                const bool stratified = true,
                                const unsigned int alignment = 0)
        {
            cv_nfolds_ = nfolds;
            cv_stratified_ = stratified;
            cv_alignment_ = alignment;
            return this;
        }
        inline Control* tune_et(const unsigned int nstages)
        {
            et_nstages_ = nstages;
            return this;
        }

    };

}


#endif /* ABCLASS_CONTROL_H */
