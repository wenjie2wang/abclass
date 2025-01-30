//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2025 Eli Lilly and Company
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
#include "MarginLoss.h"
#include "utils.h"

namespace abclass
{

    class Boost : public MarginLoss
    {
    protected:
        // cache
        double exp_inner_max_;
        double inner_min_ { - 5.0 };
        double inner_min_p1_ { - 4.0 };

    public:
        using MarginLoss::loss;

        Boost()
        {
            set_inner_min(inner_min_);
        }

        explicit Boost(const double inner_min)
        {
            set_inner_min(inner_min);
        }
        // loss function
        inline double loss(const double u) const override
        {
            if (u < inner_min_) {
                return (inner_min_p1_ - u) * exp_inner_max_;
            }
            return std::exp(- u);
        }

        // the first derivative of the loss function
        inline double dloss_du(const double u) const override
        {
            if (u < inner_min_) {
                return - exp_inner_max_;
            }
            return - std::exp(- u);
        }

        // MM lowerbound factor
        inline double mm_lowerbound() const
        {
            return exp_inner_max_;
        }

        // setter
        inline Boost* set_inner_min(const double inner_min)
        {
            if (is_gt(inner_min, 0.0)) {
                throw std::range_error("The 'inner_min' cannot be positive.");
            }
            inner_min_ = inner_min;
            inner_min_p1_ = 1.0 + inner_min_;
            exp_inner_max_ = std::exp(- inner_min_);
            return this;
        }

    };

}


#endif /* ABCLASS_BOOST_H */
