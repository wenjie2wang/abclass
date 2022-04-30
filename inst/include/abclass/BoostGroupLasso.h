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

#ifndef ABCLASS_BOOST_GROUP_LASSO_H
#define ABCLASS_BOOST_GROUP_LASSO_H

#include <RcppArmadillo.h>
#include <stdexcept>
#include "AbclassGroupLasso.h"
#include "utils.h"

namespace abclass
{
    // define class for inputs and outputs
    class BoostGroupLasso : public AbclassGroupLasso
    {
    private:
        // cache
        double exp_inner_max_;

    protected:

        double inner_min_ = - 5.0;

        // set CMD lowerbound
        inline void set_gmd_lowerbound() override
        {
            arma::mat sqx { arma::square(x_) };
            sqx.each_col() %= obs_weight_;
            gmd_lowerbound_ = exp_inner_max_ * arma::sum(sqx, 0) / dn_obs_;
        }

        // objective function without regularization
        inline double objective0(const arma::vec& inner) const override
        {
            arma::vec tmp { arma::zeros(inner.n_elem) };
            double tmp1 { 1 + inner_min_ };
            for (size_t i {0}; i < inner.n_elem; ++i) {
                if (inner[i] < inner_min_) {
                    tmp[i] = (tmp1 - inner[i]) * exp_inner_max_;
                } else {
                    tmp[i] = std::exp(- inner[i]);
                }
            }
            return arma::mean(obs_weight_ % tmp);
        }

        // the first derivative of the loss function
        inline arma::vec loss_derivative(const arma::vec& u) const override
        {
            arma::vec out { arma::zeros(u.n_elem) };
            for (size_t i {0}; i < u.n_elem; ++i) {
                if (u[i] < inner_min_) {
                    out[i] = - exp_inner_max_;
                } else {
                    out[i] = - std::exp(- u[i]);
                }
            }
            return out;
        }

    public:

        // inherit constructors
        using AbclassGroupLasso::AbclassGroupLasso;

        //! @param x The design matrix without an intercept term.
        //! @param y The category index vector.
        BoostGroupLasso(const arma::mat& x,
                        const arma::uvec& y,
                        const bool intercept = true,
                        const bool standardize = true,
                        const arma::vec& weight = arma::vec()) :
            AbclassGroupLasso(x, y, intercept, standardize, weight)
        {
            set_inner_min(- 5.0);
        }

        BoostGroupLasso* set_inner_min(const double inner_min)
        {
            if (is_gt(inner_min, 0.0)) {
                throw std::range_error("The 'inner_min' cannot be positive.");
            }
            inner_min_ = inner_min;
            exp_inner_max_ = std::exp(- inner_min_);
            return this;
        }


    };                          // end of class

}

#endif
