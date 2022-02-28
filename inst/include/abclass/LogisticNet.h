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

#ifndef ABCLASS_LOGISTIC_NET_H
#define ABCLASS_LOGISTIC_NET_H

#include <RcppArmadillo.h>
#include "AbclassNet.h"

namespace abclass
{
    // define class for inputs and outputs
    class LogisticNet : public AbclassNet
    {
    protected:

        // set CMD lowerbound
        inline void set_cmd_lowerbound() override
        {
            if (standardize_) {
                cmd_lowerbound_ = arma::ones<arma::rowvec>(p1_);
                cmd_lowerbound_ *= arma::mean(obs_weight_) / 4.0;
            } else {
                arma::mat sqx { arma::square(x_) };
                sqx.each_col() %= obs_weight_;
                cmd_lowerbound_ = arma::sum(sqx, 0) / (4.0 * dn_obs_);
            }
        }

        // objective function without regularization
        inline double objective0(const arma::vec& inner) const override
        {
            return arma::mean(obs_weight_ %
                              arma::log(1.0 + arma::exp(- inner)));
        }

        // the first derivative of the loss function
        inline arma::vec neg_loss_derivative(const arma::vec& u) const override
        {
            return 1.0 / (1.0 + arma::exp(u));
        }

    public:

        // inherit constructors
        using AbclassNet::AbclassNet;

        //! @param x The design matrix without an intercept term.
        //! @param y The category index vector.
        LogisticNet(const arma::mat& x,
                    const arma::uvec& y,
                    const bool intercept = true,
                    const bool standardize = true,
                    const arma::vec& weight = arma::vec()) :
            AbclassNet(x, y, intercept, standardize, weight)
        {
            // set the CMD lowerbound (which needs to be done only once)
            // set_cmd_lowerbound();
        }


    };                          // end of class

}

#endif
