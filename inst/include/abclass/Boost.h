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

#ifndef ABCLASS_BOOST_H
#define ABCLASS_BOOST_H

#include <RcppArmadillo.h>
#include <stdexcept>
#include "utils.h"

namespace abclass
{

    class Boost
    {
    protected:
        // cache
        double exp_inner_max_;
        double inner_min_ = - 5.0;

    public:
        Boost()
        {
            set_inner_min(inner_min_);
        }

        explicit Boost(const double inner_min)
        {
            set_inner_min(inner_min);
        }

        // loss function
        inline double loss(const arma::vec& u,
                           const arma::vec& obs_weight) const
        {
            arma::vec tmp { arma::zeros(u.n_elem) };
            double tmp1 { 1 + inner_min_ };
            for (size_t i {0}; i < u.n_elem; ++i) {
                if (u[i] < inner_min_) {
                    tmp[i] = (tmp1 - u[i]) * exp_inner_max_;
                } else {
                    tmp[i] = std::exp(- u[i]);
                }
            }
            return arma::mean(obs_weight % tmp);
        }

        // the first derivative of the loss function
        inline arma::vec dloss(const arma::vec& u) const
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

        // MM lowerbound
        template <typename T>
        inline arma::rowvec mm_lowerbound(const T& x,
                                          const arma::vec& obs_weight)
        {
            T sqx { arma::square(x) };
            double dn_obs { static_cast<double>(x.n_rows) };
            return exp_inner_max_ * (obs_weight.t() * sqx) / dn_obs;

        }
        // for the intercept
        inline double mm_lowerbound(const double dn_obs,
                                    const arma::vec& obs_weight)
        {
            return exp_inner_max_ * arma::accu(obs_weight) / dn_obs;
        }

        // setter
        inline Boost* set_inner_min(const double inner_min)
        {
            if (is_gt(inner_min, 0.0)) {
                throw std::range_error("The 'inner_min' cannot be positive.");
            }
            inner_min_ = inner_min;
            exp_inner_max_ = std::exp(- inner_min_);
            return this;
        }


    };

}


#endif /* ABCLASS_BOOST_H */
