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

#include "LogisticT.h"
#include "BoostT.h"
#include "HingeBoostT.h"
#include "LumT.h"

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
    using LogisticNet = LogisticT<AbclassNet<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticSCAD = LogisticT<AbclassSCAD<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticMCP = LogisticT<AbclassMCP<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticGroupLasso = LogisticT<AbclassGroupLasso<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticGroupSCAD = LogisticT<AbclassGroupSCAD<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticGroupMCP = LogisticT<AbclassGroupMCP<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticCompMCP = LogisticT<AbclassCompMCP<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticGEL = LogisticT<AbclassGEL<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticMellowL1 = LogisticT<AbclassMellowL1<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticMellowMCP = LogisticT<AbclassMellowMCP<Logistic, T_x>, T_x>;

    // Boost
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

    // HingeBoost
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

    template<typename T_x>
    using HingeBoostGEL =
        HingeBoostT<AbclassGEL<HingeBoost, T_x>, T_x>;

    template<typename T_x>
    using HingeBoostMellowL1 =
        HingeBoostT<AbclassMellowL1<HingeBoost, T_x>, T_x>;

    template<typename T_x>
    using HingeBoostMellowMCP =
        HingeBoostT<AbclassMellowMCP<HingeBoost, T_x>, T_x>;

    // Lum
    template<typename T_x>
    using LumNet = LumT<AbclassNet<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumSCAD = LumT<AbclassSCAD<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumMCP = LumT<AbclassMCP<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumGroupLasso = LumT<AbclassGroupLasso<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumGroupSCAD = LumT<AbclassGroupSCAD<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumGroupMCP = LumT<AbclassGroupMCP<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumCompMCP = LumT<AbclassCompMCP<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumGEL = LumT<AbclassGEL<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumMellowL1 = LumT<AbclassMellowL1<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumMellowMCP = LumT<AbclassMellowMCP<Lum, T_x>, T_x>;

}  // abclass

#endif /* ABCLASS_TEMPLATE_ALIAS_H */
