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

#ifndef ABCLASS_LOGISTIC_GROUP_MCP_H
#define ABCLASS_LOGISTIC_GROUP_MCP_H

#include <RcppArmadillo.h>
#include "AbclassGroupMCP.h"
#include "Logistic.h"
#include "Control.h"

namespace abclass
{
    // define class for inputs and outputs
    template <typename T_x>
    class LogisticGroupMCP : public AbclassGroupMCP<Logistic, T_x>
    {
    public:
        // inherit
        using AbclassGroupMCP<Logistic, T_x>::AbclassGroupMCP;

    };                          // end of class

}

#endif
