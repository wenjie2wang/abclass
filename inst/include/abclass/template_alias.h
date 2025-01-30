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

#ifndef ABCLASS_TEMPLATE_ALIAS_H
#define ABCLASS_TEMPLATE_ALIAS_H

#include <RcppArmadillo.h>

#include "Logistic.h"
#include "Boost.h"
#include "HingeBoost.h"
#include "Lum.h"

#include "Mlogit.h"
#include "LikeBoost.h"
#include "LikeLogistic.h"
#include "LikeHingeBoost.h"
#include "LikeLum.h"

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
    using LogitNet = AbclassNet<Logistic, T_x>;

    template<typename T_x>
    using LogitSCAD = AbclassSCAD<Logistic, T_x>;

    template<typename T_x>
    using LogitMCP = AbclassMCP<Logistic, T_x>;

    template<typename T_x>
    using LogitGLasso = AbclassGroupLasso<Logistic, T_x>;

    template<typename T_x>
    using LogitGSCAD = AbclassGroupSCAD<Logistic, T_x>;

    template<typename T_x>
    using LogitGMCP = AbclassGroupMCP<Logistic, T_x>;

    template<typename T_x>
    using LogitCMCP = AbclassCompMCP<Logistic, T_x>;

    template<typename T_x>
    using LogitGEL = AbclassGEL<Logistic, T_x>;

    template<typename T_x>
    using LogitML1 = AbclassMellowL1<Logistic, T_x>;

    template<typename T_x>
    using LogitMMCP = AbclassMellowMCP<Logistic, T_x>;

    // Boost
    template<typename T_x>
    using BoostNet = AbclassNet<Boost, T_x>;

    template<typename T_x>
    using BoostSCAD = AbclassSCAD<Boost, T_x>;

    template<typename T_x>
    using BoostMCP = AbclassMCP<Boost, T_x>;

    template<typename T_x>
    using BoostGLasso = AbclassGroupLasso<Boost, T_x>;

    template<typename T_x>
    using BoostGSCAD = AbclassGroupSCAD<Boost, T_x>;

    template<typename T_x>
    using BoostGMCP = AbclassGroupMCP<Boost, T_x>;

    template<typename T_x>
    using BoostCMCP = AbclassCompMCP<Boost, T_x>;

    template<typename T_x>
    using BoostGEL = AbclassGEL<Boost, T_x>;

    template<typename T_x>
    using BoostML1 = AbclassMellowL1<Boost, T_x>;

    template<typename T_x>
    using BoostMMCP = AbclassMellowMCP<Boost, T_x>;

    // HingeBoost
    template<typename T_x>
    using HBoostNet = AbclassNet<HingeBoost, T_x>;

    template<typename T_x>
    using HBoostSCAD = AbclassSCAD<HingeBoost, T_x>;

    template<typename T_x>
    using HBoostMCP = AbclassMCP<HingeBoost, T_x>;

    template<typename T_x>
    using HBoostGLasso = AbclassGroupLasso<HingeBoost, T_x>;

    template<typename T_x>
    using HBoostGSCAD = AbclassGroupSCAD<HingeBoost, T_x>;

    template<typename T_x>
    using HBoostGMCP = AbclassGroupMCP<HingeBoost, T_x>;

    template<typename T_x>
    using HBoostCMCP = AbclassCompMCP<HingeBoost, T_x>;

    template<typename T_x>
    using HBoostGEL = AbclassGEL<HingeBoost, T_x>;

    template<typename T_x>
    using HBoostML1 = AbclassMellowL1<HingeBoost, T_x>;

    template<typename T_x>
    using HBoostMMCP = AbclassMellowMCP<HingeBoost, T_x>;

    // Lum
    template<typename T_x>
    using LumNet = AbclassNet<Lum, T_x>;

    template<typename T_x>
    using LumSCAD = AbclassSCAD<Lum, T_x>;

    template<typename T_x>
    using LumMCP = AbclassMCP<Lum, T_x>;

    template<typename T_x>
    using LumGLasso = AbclassGroupLasso<Lum, T_x>;

    template<typename T_x>
    using LumGSCAD = AbclassGroupSCAD<Lum, T_x>;

    template<typename T_x>
    using LumGMCP = AbclassGroupMCP<Lum, T_x>;

    template<typename T_x>
    using LumCMCP = AbclassCompMCP<Lum, T_x>;

    template<typename T_x>
    using LumGEL = AbclassGEL<Lum, T_x>;

    template<typename T_x>
    using LumML1 = AbclassMellowL1<Lum, T_x>;

    template<typename T_x>
    using LumMMCP = AbclassMellowMCP<Lum, T_x>;

    // Mlogit = LikeBoost
    template<typename T_x>
    using MlogitNet = AbclassNet<Mlogit, T_x>;

    template<typename T_x>
    using MlogitSCAD = AbclassSCAD<Mlogit, T_x>;

    template<typename T_x>
    using MlogitMCP = AbclassMCP<Mlogit, T_x>;

    template<typename T_x>
    using MlogitGLasso = AbclassGroupLasso<Mlogit, T_x>;

    template<typename T_x>
    using MlogitGSCAD = AbclassGroupSCAD<Mlogit, T_x>;

    template<typename T_x>
    using MlogitGMCP = AbclassGroupMCP<Mlogit, T_x>;

    template<typename T_x>
    using MlogitCMCP = AbclassCompMCP<Mlogit, T_x>;

    template<typename T_x>
    using MlogitGEL = AbclassGEL<Mlogit, T_x>;

    template<typename T_x>
    using MlogitML1 = AbclassMellowL1<Mlogit, T_x>;

    template<typename T_x>
    using MlogitMMCP = AbclassMellowMCP<Mlogit, T_x>;

    // LikeLogistic
    template<typename T_x>
    using LeLogitNet = AbclassNet<LikeLogistic, T_x>;

    template<typename T_x>
    using LeLogitSCAD = AbclassSCAD<LikeLogistic, T_x>;

    template<typename T_x>
    using LeLogitMCP = AbclassMCP<LikeLogistic, T_x>;

    template<typename T_x>
    using LeLogitGLasso = AbclassGroupLasso<LikeLogistic, T_x>;

    template<typename T_x>
    using LeLogitGSCAD = AbclassGroupSCAD<LikeLogistic, T_x>;

    template<typename T_x>
    using LeLogitGMCP = AbclassGroupMCP<LikeLogistic, T_x>;

    template<typename T_x>
    using LeLogitCMCP = AbclassCompMCP<LikeLogistic, T_x>;

    template<typename T_x>
    using LeLogitGEL = AbclassGEL<LikeLogistic, T_x>;

    template<typename T_x>
    using LeLogitML1 = AbclassMellowL1<LikeLogistic, T_x>;

    template<typename T_x>
    using LeLogitMMCP = AbclassMellowMCP<LikeLogistic, T_x>;

    // LikeLogistic
    template<typename T_x>
    using LeBoostNet = AbclassNet<LikeBoost, T_x>;

    template<typename T_x>
    using LeBoostSCAD = AbclassSCAD<LikeBoost, T_x>;

    template<typename T_x>
    using LeBoostMCP = AbclassMCP<LikeBoost, T_x>;

    template<typename T_x>
    using LeBoostGLasso = AbclassGroupLasso<LikeBoost, T_x>;

    template<typename T_x>
    using LeBoostGSCAD = AbclassGroupSCAD<LikeBoost, T_x>;

    template<typename T_x>
    using LeBoostGMCP = AbclassGroupMCP<LikeBoost, T_x>;

    template<typename T_x>
    using LeBoostCMCP = AbclassCompMCP<LikeBoost, T_x>;

    template<typename T_x>
    using LeBoostGEL = AbclassGEL<LikeBoost, T_x>;

    template<typename T_x>
    using LeBoostML1 = AbclassMellowL1<LikeBoost, T_x>;

    template<typename T_x>
    using LeBoostMMCP = AbclassMellowMCP<LikeBoost, T_x>;

    // LikeHingeBoost
    template<typename T_x>
    using LeHBoostNet = AbclassNet<LikeHingeBoost, T_x>;

    template<typename T_x>
    using LeHBoostSCAD = AbclassSCAD<LikeHingeBoost, T_x>;

    template<typename T_x>
    using LeHBoostMCP = AbclassMCP<LikeHingeBoost, T_x>;

    template<typename T_x>
    using LeHBoostGLasso = AbclassGroupLasso<LikeHingeBoost, T_x>;

    template<typename T_x>
    using LeHBoostGSCAD = AbclassGroupSCAD<LikeHingeBoost, T_x>;

    template<typename T_x>
    using LeHBoostGMCP = AbclassGroupMCP<LikeHingeBoost, T_x>;

    template<typename T_x>
    using LeHBoostCMCP = AbclassCompMCP<LikeHingeBoost, T_x>;

    template<typename T_x>
    using LeHBoostGEL = AbclassGEL<LikeHingeBoost, T_x>;

    template<typename T_x>
    using LeHBoostML1 = AbclassMellowL1<LikeHingeBoost, T_x>;

    template<typename T_x>
    using LeHBoostMMCP = AbclassMellowMCP<LikeHingeBoost, T_x>;

    // LikeLum
    template<typename T_x>
    using LeLumNet = AbclassNet<LikeLum, T_x>;

    template<typename T_x>
    using LeLumSCAD = AbclassSCAD<LikeLum, T_x>;

    template<typename T_x>
    using LeLumMCP = AbclassMCP<LikeLum, T_x>;

    template<typename T_x>
    using LeLumGLasso = AbclassGroupLasso<LikeLum, T_x>;

    template<typename T_x>
    using LeLumGSCAD = AbclassGroupSCAD<LikeLum, T_x>;

    template<typename T_x>
    using LeLumGMCP = AbclassGroupMCP<LikeLum, T_x>;

    template<typename T_x>
    using LeLumCMCP = AbclassCompMCP<LikeLum, T_x>;

    template<typename T_x>
    using LeLumGEL = AbclassGEL<LikeLum, T_x>;

    template<typename T_x>
    using LeLumML1 = AbclassMellowL1<LikeLum, T_x>;

    template<typename T_x>
    using LeLumMMCP = AbclassMellowMCP<LikeLum, T_x>;

}  // abclass

#endif /* ABCLASS_TEMPLATE_ALIAS_H */
