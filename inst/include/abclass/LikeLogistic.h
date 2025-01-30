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

#ifndef ABCLASS_LIKELOGISTIC_H
#define ABCLASS_LIKELOGISTIC_H

#include <RcppArmadillo.h>
#include "LikeBoost.h"

namespace abclass
{

    class LikeLogistic : public LikeBoost
    {
    public:
        LikeLogistic(){}

        inline double neg_inv_d1loss(const double u) const override
        {
            // - 1 / l'(u)
            return 1.0 + std::exp(u);
        }
        inline double neg_d2_over_d1(const double u) const override
        {
            // - l''(u) / l'(u)
            return 1.0 / (1.0 + std::exp(- u));
        }
        inline double d2_over_d1s(const double u) const override
        {
            // l''(u) / (l'(u) ^ 2)
            return std::exp(u);
        }

    };

}  // abclass

#endif /* ABCLASS_LIKELOGISTIC_H */
