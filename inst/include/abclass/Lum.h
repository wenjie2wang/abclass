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

#ifndef ABCLASS_LUM_H
#define ABCLASS_LUM_H

#include <RcppArmadillo.h>
#include <stdexcept>
#include "MarginLoss.h"
#include "utils.h"

namespace abclass
{

    class Lum : public MarginLoss
    {
    private:
        // cache
        double lum_ap1_;        // a + 1
        double lum_log_a_;      // log(a)
        double lum_a_log_a_;    // a log(a)
        double lum_cp1_;        // c + 1
        double lum_log_cp1_;    // log(c + 1)
        double lum_c_cp1_;      // c / (c + 1)
        double lum_amc_;        // a - c
        double lum_mm_;         // (a + 1)(c + 1) / a
        double lum_loss_const_; // - log(c + 1) + a log(a)
        double lum_d1_const_;   // (a + 1) log(a)

    protected:
        double lum_c_ { 0.0 };  // c
        double lum_a_ { 1.0 };  // a

    public:
        using MarginLoss::loss;

        Lum()
        {
            set_ac(1.0, 0.0);
        }

        Lum(const double a, const double c)
        {
            set_ac(a, c);
        }

        // loss function
        inline double loss(const double u) const override
        {
            if (u < lum_c_cp1_) {
                return 1.0 - u;
            }
            return std::exp(lum_loss_const_ -
                            lum_a_ * std::log(lum_cp1_ * u + lum_amc_));
        }

        // the first derivative of the loss function
        inline double dloss_du(const double u) const override
        {
            if (u < lum_c_cp1_) {
                return - 1.0;
            }
            return - std::exp(lum_d1_const_ -
                              lum_ap1_ * std::log(lum_cp1_ * u + lum_amc_));
        }

        // MM lowerbound factor
        inline double mm_lowerbound() const
        {
            return lum_mm_;
        }

        inline Lum* set_ac(const double lum_a, const double lum_c)
        {
            if (is_le(lum_a, 0.0)) {
                throw std::range_error("The LUM 'a' must be positive.");
            }
            if (is_lt(lum_c, 0.0)) {
                throw std::range_error("The LUM 'c' cannot be negative.");
            }
            lum_a_ = lum_a;
            lum_ap1_ = lum_a_ + 1.0;
            lum_log_a_ = std::log(lum_a_);
            lum_a_log_a_ = lum_a_ * lum_log_a_;
            lum_c_ = lum_c;
            lum_cp1_ = lum_c + 1.0;
            lum_log_cp1_ = std::log(lum_cp1_);
            lum_c_cp1_ = 1.0 - 1.0 / lum_cp1_;
            lum_amc_ = lum_a_ - lum_c_;
            lum_mm_ = lum_ap1_ / lum_a_ * lum_cp1_;
            lum_loss_const_ = - lum_log_cp1_ + lum_a_log_a_;
            lum_d1_const_ = lum_a_log_a_ + lum_log_a_;
            return this;
        }

    };


}  // abclass

#endif /* ABCLASS_LUM_H */
