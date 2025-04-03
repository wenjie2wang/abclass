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
#include "abclass/AbclassBlockCD.h"
#include "abclass/AbclassCD.h"
#include "abclass/AbclassLinear.h"
#include "abclass/Control.h"
#include "abclass/Data.h"
#include "abclass/Mellowmax.h"

// losses
#include "abclass/Boost.h"
#include "abclass/HingeBoost.h"
#include "abclass/LikeBoost.h"
#include "abclass/LikeHingeBoost.h"
#include "abclass/LikeLogistic.h"
#include "abclass/LikeLum.h"
#include "abclass/Logistic.h"
#include "abclass/Lum.h"
#include "abclass/MarginLoss.h"
#include "abclass/Mlogit.h"

// penalties
#include "abclass/AbclassCompMCP.h"
#include "abclass/AbclassGEL.h"
#include "abclass/AbclassGroupLasso.h"
#include "abclass/AbclassGroupMCP.h"
#include "abclass/AbclassGroupSCAD.h"
#include "abclass/AbclassMCP.h"
#include "abclass/AbclassMellowL1.h"
#include "abclass/AbclassMellowMCP.h"
#include "abclass/AbclassNet.h"
#include "abclass/AbclassSCAD.h"

// aliases
#include "abclass/template_alias.h"

// utils
#include "abclass/utils.h"

// for tuning
#include "abclass/CrossValidation.h"
#include "abclass/template_cv.h"
#include "abclass/template_et.h"

namespace abclass
{

    // for a linear fit
    template <typename T, typename T_x>
    inline Output<T_x> template_linear_fit(T& object,
                                           const Data<T_x>& train_data)
    {
        object.set_data(train_data);
        if (object.control_.et_nstages_ > 0) {
            // et procedure
            et_lambda(object);
            // add estimates from cv
            if (object.control_.cv_nfolds_ > 0) {
                et_cv_accuracy(object);
            }
            return object.output_;
        }
        // main fit
        object.fit();
        // add cv results
        if (object.control_.cv_nfolds_ > 0) {
            cv_lambda(object);
        }
        return object.output_;
    }

    template <typename T>
    inline Output<T>
    abclass_linear_fit(const Data<T>& train_data, const Control& ctrl,
                       const size_t loss_id, const size_t penalty_id)
    {
        switch (loss_id) {
        case 1: { // logistic
            switch (penalty_id) {
            case 1: { // lasso
                LogitNet<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 2: { // scad
                LogitSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 3: { // mcp
                LogitMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 4: { // group lasso
                LogitGLasso<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 5: { // group scad
                LogitGSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 6: { // group mcp
                LogitGMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 7: { // composite mcp
                LogitCMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 8: { // gel
                LogitGEL<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 9: { // mellowmax L1
                LogitML1<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 10: { // mellowmax mcp
                LogitMMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            }
        }
        case 2: { // boost
            switch (penalty_id) {
            case 1: { // lasso
                BoostNet<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            case 2: { // scad
                BoostSCAD<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            case 3: { // mcp
                BoostMCP<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            case 4: { // group lasso
                BoostGLasso<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            case 5: { // group scad
                BoostGSCAD<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            case 6: { // group mcp
                BoostGMCP<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            case 7: { // composite mcp
                BoostCMCP<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            case 8: { // gel
                BoostGEL<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            case 9: { // mellowmax L1
                BoostML1<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            case 10: { // mellowmax mcp
                BoostMMCP<T> object{ctrl};
                object.loss_fun_.set_inner_min(ctrl.boost_umin_);
                return template_linear_fit(object, train_data);
            }
            }
        }
        case 3: { // hinge.boost
            switch (penalty_id) {
            case 1: { // lasso
                HBoostNet<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 2: { // scad
                HBoostSCAD<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 3: { // mcp
                HBoostMCP<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 4: { // group lasso
                HBoostGLasso<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 5: { // group scad
                HBoostGSCAD<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 6: { // group mcp
                HBoostGMCP<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 7: { // composite mcp
                HBoostCMCP<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 8: { // gel
                HBoostGEL<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 9: { // mellowmax L1
                HBoostML1<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 10: { // mellowmax mcp
                HBoostMMCP<T> object{ctrl};
                object.loss_fun_.set_c(ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            }
        }
        case 4: { // lum
            switch (penalty_id) {
            case 1: { // lasso
                LumNet<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 2: { // scad
                LumSCAD<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 3: { // mcp
                LumMCP<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 4: { // group lasso
                LumGLasso<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 5: { // group scad
                LumGSCAD<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 6: { // group mcp
                LumGMCP<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 7: { // composite mcp
                LumCMCP<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 8: { // gel
                LumGEL<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 9: { // mellowmax L1
                LumML1<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            case 10: { // mellowmax mcp
                LumMMCP<T> object{ctrl};
                object.loss_fun_.set_ac(ctrl.lum_a_, ctrl.lum_c_);
                return template_linear_fit(object, train_data);
            }
            }
        }
        case 5: { // mlogit
            switch (penalty_id) {
            case 1: { // lasso
                MlogitNet<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 2: { // scad
                MlogitSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 3: { // mcp
                MlogitMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 4: { // group lasso
                MlogitGLasso<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 5: { // group scad
                MlogitGSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 6: { // group mcp
                MlogitGMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 7: { // composite mcp
                MlogitCMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 8: { // gel
                MlogitGEL<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 9: { // mellowmax L1
                MlogitML1<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 10: { // mellowmax mcp
                MlogitMMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            }
        }
        case 6: { // LikeLogistic
            switch (penalty_id) {
            case 1: { // lasso
                LeLogitNet<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 2: { // scad
                LeLogitSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 3: { // mcp
                LeLogitMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 4: { // group lasso
                LeLogitGLasso<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 5: { // group scad
                LeLogitGSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 6: { // group mcp
                LeLogitGMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 7: { // composite mcp
                LeLogitCMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 8: { // gel
                LeLogitGEL<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 9: { // mellowmax L1
                LeLogitML1<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 10: { // mellowmax mcp
                LeLogitMMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            }
        }
        case 7: { // LikeBoost
            switch (penalty_id) {
            case 1: { // lasso
                LeBoostNet<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 2: { // scad
                LeBoostSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 3: { // mcp
                LeBoostMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 4: { // group lasso
                LeBoostGLasso<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 5: { // group scad
                LeBoostGSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 6: { // group mcp
                LeBoostGMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 7: { // composite mcp
                LeBoostCMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 8: { // gel
                LeBoostGEL<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 9: { // mellowmax L1
                LeBoostML1<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 10: { // mellowmax mcp
                LeBoostMMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            }
        }
        case 8: { // LikeHingeBoost
            switch (penalty_id) {
            case 1: { // lasso
                LeHBoostNet<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 2: { // scad
                LeHBoostSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 3: { // mcp
                LeHBoostMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 4: { // group lasso
                LeHBoostGLasso<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 5: { // group scad
                LeHBoostGSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 6: { // group mcp
                LeHBoostGMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 7: { // composite mcp
                LeHBoostCMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 8: { // gel
                LeHBoostGEL<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 9: { // mellowmax L1
                LeHBoostML1<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 10: { // mellowmax mcp
                LeHBoostMMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            }
        }
        case 9: { // LikeLum
            switch (penalty_id) {
            case 1: { // lasso
                LeLumNet<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 2: { // scad
                LeLumSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 3: { // mcp
                LeLumMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 4: { // group lasso
                LeLumGLasso<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 5: { // group scad
                LeLumGSCAD<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 6: { // group mcp
                LeLumGMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 7: { // composite mcp
                LeLumCMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 8: { // gel
                LeLumGEL<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 9: { // mellowmax L1
                LeLumML1<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            case 10: { // mellowmax mcp
                LeLumMMCP<T> object{ctrl};
                return template_linear_fit(object, train_data);
            }
            }
        }
        default:
            break;
        }
        throw std::invalid_argument("Invalid choice of loss or penalty.");
    }

} // namespace abclass

#endif
