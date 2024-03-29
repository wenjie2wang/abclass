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
#include "utils.h"

namespace abclass
{

    class Logistic
    {
    public:
        Logistic(){}

        // loss function
        inline double loss(const double u) const
        {
            return std::log(1.0 + std::exp(- u));
        }
        inline double loss(const arma::vec& u,
                           const arma::vec& obs_weight) const
        {
            double res { 0.0 };
            for (size_t i {0}; i < u.n_elem; ++i) {
                res += obs_weight(i) * loss(u[i]);
            }
            return res;
        }

        // the first derivative of the loss function
        inline double dloss(const double u) const
        {
            return - 1.0 / (1.0 + std::exp(u));
        }
        inline arma::vec dloss(const arma::vec& u) const
        {
            arma::vec out { arma::zeros(u.n_elem) };
            for (size_t i {0}; i < out.n_elem; ++i) {
                out[i] = dloss(u[i]);
            }
            return out;
            // return - 1.0 / (1.0 + arma::exp(u));
        }

        // MM lowerbound
        template <typename T>
        inline arma::rowvec mm_lowerbound(const T& x,
                                          const arma::vec& obs_weight)
        {
            T sqx { arma::square(x) };
            double dn_obs { static_cast<double>(x.n_rows) };
            return obs_weight.t() * sqx / (4.0 * dn_obs);
        }
        // for the intercept
        inline double mm_lowerbound(const double dn_obs,
                                    const arma::vec& obs_weight)
        {
            return arma::accu(obs_weight) / (4.0 * dn_obs);
        }

    };

}  // abclass

#endif /* ABCLASS_LOGISTIC_H */
