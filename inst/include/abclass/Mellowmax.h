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

#ifndef ABCLASS_MELLOWMAX_H
#define ABCLASS_MELLOWMAX_H

#include <RcppArmadillo.h>

namespace abclass
{
    class Mellowmax
    {
    protected:
        // data
        double omega_;
        arma::rowvec theta_;

        // cache
        double dn_x_;            // double(x.n_elem)
        double max_x_;           // max(x)
        double sum_exp_x_max_;   // sum(exp(x - max(x)))
        arma::rowvec x_;         // x = omega * abs(theta)
        arma::rowvec exp_x_max_; // exp(x - max(x))

    public:

        // omega != 0
        Mellowmax(const arma::rowvec& theta, const double omega)
        {
            theta_ = theta;
            omega_ = omega;
            x_ = omega_ * arma::abs(theta_);
            max_x_ = x_.max();
            dn_x_ = static_cast<double>(x_.n_elem);
            exp_x_max_ = arma::zeros<arma::rowvec>(x_.n_elem);
            // For IEEE-compatible type double,
            // overflow is guaranteed if 709.8 < num,
            // and underflow is guaranteed if num < -708.4.
            for (size_t i {0}; i < x_.n_elem; ++i) {
                const double di { x_(i) - max_x_ };
                if (di > - 500) {
                    exp_x_max_(i) = std::exp(di);
                }
            }
            sum_exp_x_max_ = arma::accu(exp_x_max_);
        }

        // Mellowmax
        inline double value() const
        {
            return (std::log(sum_exp_x_max_ / dn_x_) + max_x_) / omega_;
        }

        // gradient of Mellowmax penalty wrt the theta vector
        inline arma::rowvec grad() const
        {
            return exp_x_max_ / sum_exp_x_max_;
        }

    };

}  // abclass

#endif /* ABCLASS_MELLOWMAX_H */
