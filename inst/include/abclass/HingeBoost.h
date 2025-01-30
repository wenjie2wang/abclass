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

#ifndef ABCLASS_HINGE_BOOST_H
#define ABCLASS_HINGE_BOOST_H

#include <RcppArmadillo.h>
#include <stdexcept>
#include "MarginLoss.h"
#include "utils.h"

namespace abclass
{

    class HingeBoost : public MarginLoss
    {
    private:
        // cache
        double lum_cp1_;
        double lum_c_cp1_;

    protected:
        double lum_c_ { 0.0 };

    public:
        using MarginLoss::loss;

        HingeBoost()
        {
            set_c(0.0);
        }

        explicit HingeBoost(const double lum_c)
        {
            set_c(lum_c);
        }

        // loss function
        inline double loss(const double u) const override
        {
            if (u < lum_c_cp1_) {
                return 1.0 - u;
            }
            return std::exp(- (lum_cp1_ * u - lum_c_)) / lum_cp1_;
        }

        // the first derivative of the loss function
        inline double dloss_du(const double u) const override
        {
            if (u < lum_c_cp1_) {
                return - 1.0;
            }
            return - std::exp(- (lum_cp1_ * u - lum_c_));
        }

        // MM lowerbound factor
        inline double mm_lowerbound() const
        {
            return lum_cp1_;
        }

        // setter
        inline HingeBoost* set_c(const double lum_c)
        {
            if (is_lt(lum_c, 0.0)) {
                throw std::range_error("The LUM 'C' cannot be negative.");
            }
            lum_c_ = lum_c;
            lum_cp1_ = lum_c + 1.0;
            lum_c_cp1_ = 1.0 - 1.0 / lum_cp1_;
            return this;
        }


    };

}


#endif /* ABCLASS_HINGE_BOOST_H */
