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

#ifndef ABCLASS_ABCLASS_GROUP_LASSO_H
#define ABCLASS_ABCLASS_GROUP_LASSO_H

#include <RcppArmadillo.h>
#include "AbclassBlockCD.h"

namespace abclass
{
    template <typename T_loss, typename T_x>
    class AbclassGroupLasso : public AbclassBlockCD<T_loss, T_x>
    {
    protected:

        // the strong rule
        inline double strong_rule_rhs(const double next_lambda,
                                      const double last_lambda) const override
        {
            return 2 * next_lambda - last_lambda;
        }

    public:
        // inherit constructors
        using AbclassBlockCD<T_loss, T_x>::AbclassBlockCD;

    };
}


#endif /* ABCLASS_ABCLASS_GROUP_LASSO_H */
