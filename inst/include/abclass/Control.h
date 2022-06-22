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
        bool intercept_ { true };               // if to contrain intercepts
        arma::vec obs_weight_ { arma::vec() } ; // observational weights

        // regularization
        arma::vec lambda_  { arma::vec() };
        unsigned int nlambda_ { 20 };
        double lambda_min_ratio_ { 0.01 };
        //   elastic-net
        double alpha_ { 0.5 };
        //   group {lasso,scad,mcp}
        arma::vec group_weight_ { arma::vec() }; // adaptive group weights
        //   group {scad,mcp}
        double dgamma_ { 0.01 }; // delta gamma
        double gamma_ { - 1.0 }; // gamma for group ncv penalty

        // tuning
        //   cross-validation
        unsigned int cv_nfolds_ { 0 };
        bool cv_stratified_ { true };
        unsigned int cv_alignment_ { 0 };
        //   ET-lasso
        unsigned int et_nstages_ { 0 };

        // optimization
        unsigned int max_iter_ { 100000 }; // maximum number of iterations
        double epsilon_ { 1e-3 };          // tolerance to check convergence
        bool varying_active_set_ { true }; // if active set should be adaptive
        bool standardize_ { true };        // is x_ standardized (column-wise)
        unsigned int verbose_ { 0 };

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
        Control* set_intercept(const bool intercept)
        {
            intercept_ = intercept;
            return this;
        }
        Control* set_weight(const arma::vec& obs_weight)
        {
            obs_weight_ = obs_weight;
            return this;
        }
        Control* set_standardize(const bool standardize)
        {
            standardize_ = standardize;
            return this;
        }
        Control* set_verbose(const unsigned int verbose)
        {
            verbose_ = verbose;
            return this;
        }
        // regularization
        Control* reg_path(const unsigned int nlambda,
                          const double lambda_min_ratio,
                          const bool varying_active_set)
        {
            if (is_le(lambda_min_ratio, 0.0)) {
                throw std::range_error(
                    "The 'lambda_min_ratio' must be positive.");
            }
            lambda_min_ratio_ = lambda_min_ratio;
            nlambda_ = nlambda;
            varying_active_set_ = varying_active_set;
            return this;
        }
        Control* reg_path(const arma::vec& lambda)
        {
            lambda_ = lambda;
            return this;
        }
        Control* reg_net(const double alpha)
        {
            // check alpha
            if ((alpha < 0.0) || (alpha > 1.0)) {
                throw std::range_error("The 'alpha' must be between 0 and 1.");
            }
            alpha_ = alpha;
            return this;
        }
        Control* reg_group(const arma::vec& group_weight,
                           const double dgamma = 1.0)
        {
            group_weight_ = group_weight;
            if (dgamma > 0.0) {
                dgamma_ = dgamma;
            } else {
                throw std::range_error("The 'dgamma' must be positive.");
            }
            return this;
        }
        // tuning
        Control* tune_cv(const unsigned int nfolds,
                         const bool stratified = true,
                         const unsigned int alignment = 0)
        {
            cv_nfolds_ = nfolds;
            cv_stratified_ = stratified;
            cv_alignment_ = alignment;
            return this;
        }
        Control* tune_et(const unsigned int nstages)
        {
            et_nstages_ = nstages;
            return this;
        }


    };

}


#endif /* ABCLASS_CONTROL_H */
