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

#ifndef ABCLASS_H
#define ABCLASS_H

#ifndef ARMA_NO_DEBUG
#define ARMA_NO_DEBUG
#endif

// classes
#include "abclass/Abclass.h"
#include "abclass/AbclassLinear.h"
#include "abclass/AbclassCD.h"
#include "abclass/AbclassBlockCD.h"
#include "abclass/Control.h"
#include "abclass/Simplex.h"
#include "abclass/Mellowmax.h"

// losses
#include "abclass/MarginLoss.h"
#include "abclass/Boost.h"
#include "abclass/Logistic.h"
#include "abclass/HingeBoost.h"
#include "abclass/Lum.h"
#include "abclass/Mlogit.h"
#include "abclass/LikeBoost.h"
#include "abclass/LikeLogistic.h"
#include "abclass/LikeHingeBoost.h"
#include "abclass/LikeLum.h"

// penalties
#include "abclass/AbclassNet.h"
#include "abclass/AbclassSCAD.h"
#include "abclass/AbclassMCP.h"
#include "abclass/AbclassGroupLasso.h"
#include "abclass/AbclassGroupSCAD.h"
#include "abclass/AbclassGroupMCP.h"
#include "abclass/AbclassCompMCP.h"
#include "abclass/AbclassGEL.h"
#include "abclass/AbclassMellowL1.h"
#include "abclass/AbclassMellowMCP.h"

// aliases
#include "abclass/template_alias.h"

// utils
#include "abclass/utils.h"

// for tuning
#include "abclass/CrossValidation.h"
#include "abclass/template_cv.h"
#include "abclass/template_et.h"

#endif
