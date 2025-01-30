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

#ifndef ABCLASS_LIKELUM_H
#define ABCLASS_LIKELUM_H

#include <RcppArmadillo.h>
#include <stdexcept>
#include "utils.h"
#include "LikeBoost.h"

namespace abclass
{

    class LikeLum : public LikeBoost
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
        LikeLum()
        {
            set_ac(1.0, 0.0);
        }

        explicit LikeLum(const double a, const double c)
        {
            set_ac(1.0, 0.0);
        }

        // setter
        inline LikeLum* set_ac(const double lum_a, const double lum_c)
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

        inline double neg_inv_d1loss(const double u) const override
        {
            // - 1 / l'(u)
            if (u < lum_c_cp1_) {
                return 1.0;
            }
            return std::exp(- lum_d1_const_ +
                            lum_ap1_ * std::log(lum_cp1_ * u + lum_amc_));
        }
        inline double neg_d2_over_d1(const double u) const override
        {
            // - l''(u) / l'(u)
            if (u < lum_c_cp1_) {
                return 0.0;
            }
            return lum_ap1_  / (u + lum_amc_ / lum_cp1_);
        }
        inline double d2_over_d1s(const double u) const override
        {
            // l''(u) / (l'(u) ^ 2)
            if (u < lum_c_cp1_) {
                return 0.0;
            }
            return lum_ap1_  / (u + lum_amc_ / lum_cp1_) *
                std::exp(- lum_d1_const_ +
                         lum_ap1_ * std::log(lum_cp1_ * u + lum_amc_));
        }

        // MM lowerbound factor
        inline double mm_lowerbound(const double dk) const override
        {
            return lum_ap1_ * lum_cp1_ / lum_a_;
        }

    };

}  // abclass

#endif /* ABCLASS_LIKELUM_H */
