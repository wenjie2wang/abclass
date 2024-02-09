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

#ifndef ABCLASS_BOOST_T_H
#define ABCLASS_BOOST_T_H

#include <RcppArmadillo.h>
#include "Boost.h"
#include "Control.h"
#include "AbclassNet.h"
#include "AbclassSCAD.h"
#include "AbclassMCP.h"
#include "AbclassGroupLasso.h"
#include "AbclassGroupSCAD.h"
#include "AbclassGroupMCP.h"
#include "AbclassCompMCP.h"
#include "AbclassGEL.h"
#include "AbclassMellowL1.h"
#include "AbclassMellowMCP.h"

namespace abclass
{
    // define class for inputs and outputs
    template <typename T_class, typename T_x>
    class BoostT : public T_class
    {
    public:
        // inherit constructors
        using T_class::T_class;

        BoostT(const T_x& x,
               const arma::uvec& y,
               const Control& control) :
            T_class(x, y, control)
        {
            this->loss_fun_.set_inner_min(- 5.0);
        }

    };                          // end of class

    // alias templates
    template<typename T_x>
    using BoostNet = BoostT<AbclassNet<Boost, T_x>, T_x>;

    template<typename T_x>
    using BoostSCAD = BoostT<AbclassSCAD<Boost, T_x>, T_x>;

    template<typename T_x>
    using BoostMCP = BoostT<AbclassMCP<Boost, T_x>, T_x>;

    template<typename T_x>
    using BoostGroupLasso = BoostT<AbclassGroupLasso<Boost, T_x>, T_x>;

    template<typename T_x>
    using BoostGroupSCAD = BoostT<AbclassGroupSCAD<Boost, T_x>, T_x>;

    template<typename T_x>
    using BoostGroupMCP = BoostT<AbclassGroupMCP<Boost, T_x>, T_x>;

    template<typename T_x>
    using BoostCompMCP = BoostT<AbclassCompMCP<Boost, T_x>, T_x>;

    template<typename T_x>
    using BoostGEL = BoostT<AbclassGEL<Boost, T_x>, T_x>;

    template<typename T_x>
    using BoostMellowL1 = BoostT<AbclassMellowL1<Boost, T_x>, T_x>;

    template<typename T_x>
    using BoostMellowMCP = BoostT<AbclassMellowMCP<Boost, T_x>, T_x>;

}

#endif
