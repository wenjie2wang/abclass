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

#ifndef ABCLASS_ABCLASS_GROUP_LASSO_H
#define ABCLASS_ABCLASS_GROUP_LASSO_H

#include <RcppArmadillo.h>
#include "AbclassGroup.h"
#include "Control.h"
#include "utils.h"

namespace abclass
{
    template <typename T_loss, typename T_x>
    class AbclassGroupLasso : public AbclassGroup<T_loss, T_x>
    {
    protected:
        // data
        using AbclassGroup<T_loss, T_x>::inter_;

        // GMD update step for beta
        inline void update_beta_g(arma::mat::row_iterator beta_g_it,
                                  const arma::rowvec& u_g,
                                  const double l1_lambda_g,
                                  const double l2_lambda,
                                  const double m_g) override
        {
            arma::rowvec z_mg { u_g };
            for (size_t i {0}; i < z_mg.n_elem; ++i) {
                z_mg[i] += m_g * *(std::next(beta_g_it, i));;
            }
            const double pos_part { 1.0 - l1_lambda_g / l2_norm(z_mg) };
            if (pos_part > 0.0) {
                for (size_t i {0}; i < z_mg.n_elem; ++beta_g_it, ++i) {
                    *beta_g_it = z_mg[i] * pos_part / (m_g + l2_lambda);
                }
            } else {
                for (size_t i {0}; i < z_mg.n_elem; ++beta_g_it, ++i) {
                    *beta_g_it = 0.0;
                }
            }
        }

        // the strong rule
        inline double strong_rule_rhs(const double next_lambda,
                                      const double last_lambda) const override
        {
            return 2 * next_lambda - last_lambda;
        }

    public:
        // inherit constructors
        using AbclassGroup<T_loss, T_x>::AbclassGroup;

        // data members
        using AbclassGroup<T_loss, T_x>::control_;

    };
}


#endif /* ABCLASS_ABCLASS_GROUP_LASSO_H */
