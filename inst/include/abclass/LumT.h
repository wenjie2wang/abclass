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

#ifndef ABCLASS_LUM_T_H
#define ABCLASS_LUM_T_H

#include <RcppArmadillo.h>
#include "AbclassNet.h"
#include "AbclassGroupLasso.h"
#include "AbclassGroupSCAD.h"
#include "AbclassGroupMCP.h"
#include "Lum.h"
#include "Control.h"

namespace abclass
{
    template <typename T_class, typename T_x>
    class LumT : public T_class
    {
    public:
        // inherit constructors
        using T_class::T_class;

        LumT(const T_x& x,
             const arma::uvec& y,
             const Control& control) :
            T_class(x, y, control)
        {
            this->loss_.set_ac(1.0, 0.0);
        }

    };                          // end of class

    // alias templates
    template<typename T_x>
    using LumNet = LumT<AbclassNet<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumGroupLasso = LumT<AbclassGroupLasso<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumGroupSCAD = LumT<AbclassGroupSCAD<Lum, T_x>, T_x>;

    template<typename T_x>
    using LumGroupMCP = LumT<AbclassGroupMCP<Lum, T_x>, T_x>;

}

#endif
