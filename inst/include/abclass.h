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

#ifndef ABCLASS_H
#define ABCLASS_H

// base class
#include "abclass/Abclass.h"

// with elastic-net
#include "abclass/AbclassNet.h"
#include "abclass/LogisticNet.h"
#include "abclass/BoostNet.h"
#include "abclass/HingeBoostNet.h"
#include "abclass/LumNet.h"

// with group lasso
#include "abclass/AbclassGroupLasso.h"
#include "abclass/LogisticGroupLasso.h"
#include "abclass/BoostGroupLasso.h"
#include "abclass/HingeBoostGroupLasso.h"
#include "abclass/LumGroupLasso.h"

// with group scad
#include "abclass/AbclassGroupSCAD.h"
#include "abclass/LogisticGroupSCAD.h"
#include "abclass/BoostGroupSCAD.h"
#include "abclass/HingeBoostGroupSCAD.h"
#include "abclass/LumGroupSCAD.h"

// with group mcp
#include "abclass/AbclassGroupMCP.h"
#include "abclass/LogisticGroupMCP.h"
#include "abclass/BoostGroupMCP.h"
#include "abclass/HingeBoostGroupMCP.h"
#include "abclass/LumGroupMCP.h"

// simplex class
#include "abclass/Simplex.h"

// utils
#include "abclass/utils.h"

// for cross-validation
#include "abclass/CrossValidation.h"
#include "abclass/template_cv.h"

#endif
