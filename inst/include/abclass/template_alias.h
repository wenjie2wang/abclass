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

#ifndef ABCLASS_TEMPLATE_ALIAS_H
#define ABCLASS_TEMPLATE_ALIAS_H

#include <RcppArmadillo.h>

#include "Logistic.h"
#include "Boost.h"
#include "HingeBoost.h"
#include "Lum.h"
#include "Mlogit.h"

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

    // Logistic
    template<typename T_x>
    using LogisticNet = AbclassNet<Logistic, T_x>;

    template<typename T_x>
    using LogisticSCAD = AbclassSCAD<Logistic, T_x>;

    template<typename T_x>
    using LogisticMCP = AbclassMCP<Logistic, T_x>;

    template<typename T_x>
    using LogisticGroupLasso = AbclassGroupLasso<Logistic, T_x>;

    template<typename T_x>
    using LogisticGroupSCAD = AbclassGroupSCAD<Logistic, T_x>;

    template<typename T_x>
    using LogisticGroupMCP = AbclassGroupMCP<Logistic, T_x>;

    template<typename T_x>
    using LogisticCompMCP = AbclassCompMCP<Logistic, T_x>;

    template<typename T_x>
    using LogisticGEL = AbclassGEL<Logistic, T_x>;

    template<typename T_x>
    using LogisticMellowL1 = AbclassMellowL1<Logistic, T_x>;

    template<typename T_x>
    using LogisticMellowMCP = AbclassMellowMCP<Logistic, T_x>;

    // Boost
    template<typename T_x>
    using BoostNet = AbclassNet<Boost, T_x>;

    template<typename T_x>
    using BoostSCAD = AbclassSCAD<Boost, T_x>;

    template<typename T_x>
    using BoostMCP = AbclassMCP<Boost, T_x>;

    template<typename T_x>
    using BoostGroupLasso = AbclassGroupLasso<Boost, T_x>;

    template<typename T_x>
    using BoostGroupSCAD = AbclassGroupSCAD<Boost, T_x>;

    template<typename T_x>
    using BoostGroupMCP = AbclassGroupMCP<Boost, T_x>;

    template<typename T_x>
    using BoostCompMCP = AbclassCompMCP<Boost, T_x>;

    template<typename T_x>
    using BoostGEL = AbclassGEL<Boost, T_x>;

    template<typename T_x>
    using BoostMellowL1 = AbclassMellowL1<Boost, T_x>;

    template<typename T_x>
    using BoostMellowMCP = AbclassMellowMCP<Boost, T_x>;

    // HingeBoost
    template<typename T_x>
    using HingeBoostNet = AbclassNet<HingeBoost, T_x>;

    template<typename T_x>
    using HingeBoostSCAD = AbclassSCAD<HingeBoost, T_x>;

    template<typename T_x>
    using HingeBoostMCP = AbclassMCP<HingeBoost, T_x>;

    template<typename T_x>
    using HingeBoostGroupLasso = AbclassGroupLasso<HingeBoost, T_x>;

    template<typename T_x>
    using HingeBoostGroupSCAD = AbclassGroupSCAD<HingeBoost, T_x>;

    template<typename T_x>
    using HingeBoostGroupMCP = AbclassGroupMCP<HingeBoost, T_x>;

    template<typename T_x>
    using HingeBoostCompMCP = AbclassCompMCP<HingeBoost, T_x>;

    template<typename T_x>
    using HingeBoostGEL = AbclassGEL<HingeBoost, T_x>;

    template<typename T_x>
    using HingeBoostMellowL1 = AbclassMellowL1<HingeBoost, T_x>;

    template<typename T_x>
    using HingeBoostMellowMCP = AbclassMellowMCP<HingeBoost, T_x>;

    // Lum
    template<typename T_x>
    using LumNet = AbclassNet<Lum, T_x>;

    template<typename T_x>
    using LumSCAD = AbclassSCAD<Lum, T_x>;

    template<typename T_x>
    using LumMCP = AbclassMCP<Lum, T_x>;

    template<typename T_x>
    using LumGroupLasso = AbclassGroupLasso<Lum, T_x>;

    template<typename T_x>
    using LumGroupSCAD = AbclassGroupSCAD<Lum, T_x>;

    template<typename T_x>
    using LumGroupMCP = AbclassGroupMCP<Lum, T_x>;

    template<typename T_x>
    using LumCompMCP = AbclassCompMCP<Lum, T_x>;

    template<typename T_x>
    using LumGEL = AbclassGEL<Lum, T_x>;

    template<typename T_x>
    using LumMellowL1 = AbclassMellowL1<Lum, T_x>;

    template<typename T_x>
    using LumMellowMCP = AbclassMellowMCP<Lum, T_x>;

    // Mlogit
    template<typename T_x>
    using MlogitNet = AbclassNet<Mlogit, T_x>;

    template<typename T_x>
    using MlogitSCAD = AbclassSCAD<Mlogit, T_x>;

    template<typename T_x>
    using MlogitMCP = AbclassMCP<Mlogit, T_x>;

    template<typename T_x>
    using MlogitGroupLasso = AbclassGroupLasso<Mlogit, T_x>;

    template<typename T_x>
    using MlogitGroupSCAD = AbclassGroupSCAD<Mlogit, T_x>;

    template<typename T_x>
    using MlogitGroupMCP = AbclassGroupMCP<Mlogit, T_x>;

    template<typename T_x>
    using MlogitCompMCP = AbclassCompMCP<Mlogit, T_x>;

    template<typename T_x>
    using MlogitGEL = AbclassGEL<Mlogit, T_x>;

    template<typename T_x>
    using MlogitMellowL1 = AbclassMellowL1<Mlogit, T_x>;

    template<typename T_x>
    using MlogitMellowMCP = AbclassMellowMCP<Mlogit, T_x>;

}  // abclass

#endif /* ABCLASS_TEMPLATE_ALIAS_H */
