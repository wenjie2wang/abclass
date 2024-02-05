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

#ifndef ABCLASS_MOML_H
#define ABCLASS_MOML_H

#include <RcppArmadillo.h>
#include "Abclass.h"
#include "AbclassNet.h"
#include "AbclassGroupLasso.h"
#include "AbclassGroupSCAD.h"
#include "AbclassGroupMCP.h"
#include "Control.h"

namespace abclass
{
    // multicategory outcome-weighted margin-based learning (MOML)
    template <typename T_class, typename T_x>
    class Moml : public T_class
    {
    protected:
        using T_class::t_vertex_;
        using T_class::km1_;

        inline void set_ex_vertex_matrix() override
        {
            ex_vertex_ = arma::mat(n_obs_, km1_);
            for (size_t i {0}; i < n_obs_; ++i) {
                if (reward_(i) < 0) {
                    ex_vertex_.row(i) = - t_vertex_.row(y_[i]);
                } else {
                    ex_vertex_.row(i) = t_vertex_.row(y_[i]);
                }
            }
        }

    public:
        // data
        using T_class::ex_vertex_;
        using T_class::n_obs_;
        using T_class::y_;      // treatment A
        using T_class::control_;

        // function
        using T_class::set_data;
        using T_class::set_weight;

        arma::vec reward_;
        arma::vec propensity_score_;

        // main constructor
        Moml(const T_x& x,
             const arma::uvec& treatment,
             const arma::vec& reward,
             const arma::vec& propensity_score,
             const Control& control = Control()) :
            reward_ (reward),
            propensity_score_ (propensity_score)
        {
            control_ = control;
            set_data(x, treatment);
            set_weight(control_.weight); // initialize weights
            set_weight(control_.weight %
                       arma::abs(reward_) / propensity_score_);
        }

    };                          // end of class

    // alias template
    template <typename T_loss, typename T_x>
    using MomlNet = Moml<AbclassNet<T_loss, T_x>, T_x>;

    template <typename T_loss, typename T_x>
    using MomlGroupLasso = Moml<AbclassGroupLasso<T_loss, T_x>, T_x>;

    template <typename T_loss, typename T_x>
    using MomlGroupSCAD = Moml<AbclassGroupSCAD<T_loss, T_x>, T_x>;

    template <typename T_loss, typename T_x>
    using MomlGroupMCP = Moml<AbclassGroupMCP<T_loss, T_x>, T_x>;

}  // abclass

#endif /* ABCLASS_MOML_H */
