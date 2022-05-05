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

#ifndef ABCLASS_LOGISTIC_GROUP_MCP_H
#define ABCLASS_LOGISTIC_GROUP_MCP_H

#include <RcppArmadillo.h>
#include "AbclassGroupMCP.h"

namespace abclass
{
    // define class for inputs and outputs
    class LogisticGroupMCP : public AbclassGroupMCP
    {
    protected:

        // set GMD lowerbound
        inline void set_gmd_lowerbound() override
        {
            arma::mat sqx { arma::square(x_) };
            sqx.each_col() %= obs_weight_;
            gmd_lowerbound_ = arma::sum(sqx, 0) / (4.0 * dn_obs_);
            max_mg_ = gmd_lowerbound_.max();
        }

        // objective function without regularization
        inline double objective0(const arma::vec& inner) const override
        {
            return arma::mean(obs_weight_ %
                              arma::log(1.0 + arma::exp(- inner)));
        }

        // the first derivative of the loss function
        inline arma::vec loss_derivative(const arma::vec& u) const override
        {
            arma::vec out { arma::zeros(u.n_elem) };
            for (size_t i {0}; i < out.n_elem; ++i) {
                out[i] = - 1.0 / (1.0 + std::exp(u[i]));
            }
            return out;
            // return - 1.0 / (1.0 + arma::exp(u));
        }

    public:

        // inherit constructors
        using AbclassGroupMCP::AbclassGroupMCP;

        //! @param x The design matrix without an intercept term.
        //! @param y The category index vector.
        LogisticGroupMCP(const arma::mat& x,
                         const arma::uvec& y,
                         const bool intercept = true,
                         const bool standardize = true,
                         const arma::vec& weight = arma::vec()) :
            AbclassGroupMCP(x, y, intercept, standardize, weight)
        {
        }


    };                          // end of class

}

#endif
