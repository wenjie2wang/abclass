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

#ifndef ABCLASS_LOGISTIC_H
#define ABCLASS_LOGISTIC_H

#include <RcppArmadillo.h>

#include "Simplex.h"
#include "MarginLoss.h"
#include "utils.h"

namespace abclass
{

    class Logistic : public MarginLoss
    {
    public:
        using MarginLoss::loss;

        Logistic(){}

        // loss function
        inline double loss(const double u) const override
        {
            return std::log(1.0 + std::exp(- u));
        }

        // the first derivative of the loss function
        inline double dloss_du(const double u) const override
        {
            return - 1.0 / (1.0 + std::exp(u));
        }

        // MM lowerbound
        template <typename T_x>
        inline arma::rowvec mm_lowerbound(
            const Simplex2<T_x>& data,
            const arma::vec& obs_weight
            ) const
        {
            T_x sqx { arma::square(data.x_) };
            return obs_weight.t() * sqx / (4.0 * data.dn_obs_);
        }

        // for the intercept
        template <typename T_x>
        inline double mm_lowerbound0(
            const Simplex2<T_x>& data,
            const arma::vec& obs_weight
            ) const
        {
            return arma::accu(obs_weight) / (4.0 * data.dn_obs_);
        }

    };

}  // abclass

#endif /* ABCLASS_LOGISTIC_H */
