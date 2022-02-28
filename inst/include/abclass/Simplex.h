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

#ifndef ABCLASS_SIMPLEX_H
#define ABCLASS_SIMPLEX_H

#include <RcppArmadillo.h>
#include <stdexcept>

namespace abclass
{

    class Simplex
    {
    private:
        unsigned int k_;        // dimensions
        // k vertex row vectors in R^(k-1) => k by (k - 1)
        arma::mat vertex_;

    public:
        // default constructor
        Simplex(const unsigned int k = 2)
        {
            if (k < 2) {
                throw std::range_error("k must be an integer > 1.");
            }
            k_ = k;
            vertex_ = arma::zeros(k_, k_ - 1);
            const arma::rowvec tmp { arma::ones<arma::rowvec>(k_ - 1) };
            vertex_.row(0) = std::pow(k_ - 1.0, - 0.5) * tmp;
            for (size_t j {1}; j < k_; ++j) {
                vertex_.row(j) = - (1.0 + std::sqrt(k_)) /
                    std::pow(k_ - 1.0, 1.5) * tmp;
                vertex_(j, j - 1) += std::sqrt(k_ / (k_ - 1.0));
            }
        }

        // setter and getter
        inline unsigned int get_k() const
        {
            return k_;
        }

        inline arma::mat get_vertex() const
        {
            return vertex_;
        }

    };

}

#endif
