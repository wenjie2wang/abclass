//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2023 Eli Lilly and Company
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

#ifndef ARMA_NO_DEBUG
#define ARMA_NO_DEBUG
#endif

// class
#include "abclass/Abclass.h"
#include "abclass/AbclassGroup.h"
#include "abclass/Control.h"
#include "abclass/Simplex.h"
#include "abclass/Moml.h"
#include "abclass/Query.h"
#include "abclass/Abrank.h"

// loss
#include "abclass/Boost.h"
#include "abclass/Logistic.h"
#include "abclass/HingeBoost.h"
#include "abclass/Lum.h"

// penalty
#include "abclass/AbclassNet.h"
#include "abclass/AbclassGroupLasso.h"
#include "abclass/AbclassGroupSCAD.h"
#include "abclass/AbclassGroupMCP.h"

// combination
#include "abclass/LogisticT.h"
#include "abclass/BoostT.h"
#include "abclass/HingeBoostT.h"
#include "abclass/LumT.h"

// utils
#include "abclass/utils.h"

// for tuning
#include "abclass/CrossValidation.h"
#include "abclass/template_cv.h"
#include "abclass/template_et.h"

#endif
