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
#include "Logistic.h"
#include "Control.h"

namespace abclass
{
    // define class for inputs and outputs
    template <typename T>
    class LogisticNet : public AbclassNet<T>, public Logistic
    {
    protected:
        using AbclassNet<T>::cmd_lowerbound_;
        using AbclassNet<T>::cmd_lowerbound0_;
        using AbclassNet<T>::dn_obs_;

        // set CMD lowerbound
        inline void set_cmd_lowerbound() override
        {
            if (control_.intercept_) {
                cmd_lowerbound0_ = Logistic::mm_lowerbound(
                    dn_obs_, control_.obs_weight_);
            }
            cmd_lowerbound_ = Logistic::mm_lowerbound(x_, control_.obs_weight_);
        }

        // objective function without regularization
        inline double objective0(const arma::vec& inner) const override
        {
            return Logistic::loss(inner, control_.obs_weight_);
        }

        // the first derivative of the loss function
        inline arma::vec loss_derivative(const arma::vec& u) const override
        {
            return Logistic::dloss(u);
        }

    public:
        // inherit
        using AbclassNet<T>::AbclassNet;
        using AbclassNet<T>::x_;
        using AbclassNet<T>::control_;

    };                          // end of class

}

#endif
