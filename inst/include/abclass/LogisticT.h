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

#ifndef ABCLASS_LOGISTIC_T_H
#define ABCLASS_LOGISTIC_T_H

#include <RcppArmadillo.h>
#include "Logistic.h"
#include "Control.h"
#include "AbclassNet.h"
#include "AbclassSCAD.h"
#include "AbclassMCP.h"
#include "AbclassGroupLasso.h"
#include "AbclassGroupSCAD.h"
#include "AbclassGroupMCP.h"
#include "AbclassCompMCP.h"
#include "AbclassGEL.h"
#include "AbclassMellowmax.h"
#include "AbclassMellowMCP.h"

namespace abclass
{
    template <typename T_class, typename T_x>
    class LogisticT : public T_class
    {
    public:
        // inherit constructors
        using T_class::T_class;
    };                          // end of class

    // alias templates
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
    using LogisticMellowmax = LogisticT<AbclassMellowmax<Logistic, T_x>, T_x>;

    template<typename T_x>
    using LogisticMellowMCP = LogisticT<AbclassMellowMCP<Logistic, T_x>, T_x>;

}

#endif
