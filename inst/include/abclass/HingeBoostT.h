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

#ifndef ABCLASS_HINGE_BOOST_T_H
#define ABCLASS_HINGE_BOOST_T_H

#include <RcppArmadillo.h>
#include "AbclassNet.h"
#include "AbclassSCAD.h"
#include "AbclassMCP.h"
#include "AbclassGroupLasso.h"
#include "AbclassGroupSCAD.h"
#include "AbclassGroupMCP.h"
#include "AbclassCompMCP.h"
#include "HingeBoost.h"
#include "Control.h"

namespace abclass
{
    // define class for inputs and outputs
    template <typename T_class, typename T_x>
    class HingeBoostT : public T_class
    {
    public:
        // inherit constructors
        using T_class::T_class;

        HingeBoostT(const T_x& x,
                    const arma::uvec& y,
                    const Control& control) :
            T_class(x, y, control)
        {
            this->loss_fun_.set_c(0.0);
        }

    };                          // end of class

    // alias templates
    template<typename T_x>
    using HingeBoostNet =
        HingeBoostT<AbclassNet<HingeBoost, T_x>, T_x>;

    template<typename T_x>
    using HingeBoostSCAD =
        HingeBoostT<AbclassSCAD<HingeBoost, T_x>, T_x>;

    template<typename T_x>
    using HingeBoostMCP =
        HingeBoostT<AbclassMCP<HingeBoost, T_x>, T_x>;

    template<typename T_x>
    using HingeBoostGroupLasso =
        HingeBoostT<AbclassGroupLasso<HingeBoost, T_x>, T_x>;

    template<typename T_x>
    using HingeBoostGroupSCAD =
        HingeBoostT<AbclassGroupSCAD<HingeBoost, T_x>, T_x>;

    template<typename T_x>
    using HingeBoostGroupMCP =
        HingeBoostT<AbclassGroupMCP<HingeBoost, T_x>, T_x>;

    template<typename T_x>
    using HingeBoostCompMCP =
        HingeBoostT<AbclassCompMCP<HingeBoost, T_x>, T_x>;

}

#endif
