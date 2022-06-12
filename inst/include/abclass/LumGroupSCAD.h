//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2022 Eli Lilly and Company
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

#ifndef ABCLASS_LUM_GROUP_SCAD_H
#define ABCLASS_LUM_GROUP_SCAD_H

#include <RcppArmadillo.h>
#include "AbclassGroupSCAD.h"
#include "Lum.h"
#include "Control.h"

namespace abclass
{
    // define class for inputs and outputs
    template <typename T_x>
    class LumGroupSCAD : public AbclassGroupSCAD<Lum, T_x>
    {
    public:
        // inherit
        using AbclassGroupSCAD<Lum, T_x>::AbclassGroupSCAD;

        LumGroupSCAD(const T_x& x,
                     const arma::uvec& y,
                     const Control& control) :
            AbclassGroupSCAD<Lum, T_x>(x, y, control)
        {
            this->loss_.set_ac(1.0, 0.0);
        }

    };                          // end of class

}

#endif
