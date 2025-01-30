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

#ifndef ABCLASS_LIKEHINGEBOOST_H
#define ABCLASS_LIKEHINGEBOOST_H

#include <RcppArmadillo.h>
#include <stdexcept>
#include "utils.h"
#include "LikeBoost.h"

namespace abclass
{

    class LikeHingeBoost : public LikeBoost
    {
    private:
        // cache
        double lum_cp1_;
        double lum_c_cp1_;

    protected:
        double lum_c_ { 0.0 };

    public:
        LikeHingeBoost()
        {
            set_c(0.0);
        }

        explicit LikeHingeBoost(const double lum_c)
        {
            set_c(lum_c);
        }

        // setter
        inline LikeHingeBoost* set_c(const double lum_c)
        {
            if (is_lt(lum_c, 0.0)) {
                throw std::range_error("The LUM 'C' cannot be negative.");
            }
            lum_c_ = lum_c;
            lum_cp1_ = lum_c + 1.0;
            lum_c_cp1_ = 1.0 - 1.0 / lum_cp1_;
            return this;
        }

        inline double neg_inv_d1loss(const double u) const override
        {
            // - 1 / l'(u)
            if (u < lum_c_cp1_) {
                return 1.0;
            }
            return std::exp(lum_cp1_ * u - lum_c_);
        }
        inline double neg_d2_over_d1(const double u) const override
        {
            // - l''(u) / l'(u)
            if (u < lum_c_cp1_) {
                return 0.0;
            }
            return lum_cp1_;
        }
        inline double d2_over_d1s(const double u) const override
        {
            // l''(u) / (l'(u) ^ 2)
            if (u < lum_c_cp1_) {
                return 0.0;
            }
            return lum_cp1_ * std::exp(lum_cp1_ * u - lum_c_);
        }

        // MM lowerbound factor
        inline double mm_lowerbound(const double dk) const override
        {
            return lum_cp1_;
        }

    };

}  // abclass

#endif /* ABCLASS_LIKEHINGEBOOST_H */
