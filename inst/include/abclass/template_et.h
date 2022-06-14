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

#ifndef ABCLASS_TEMPLATE_ET_H
#define ABCLASS_TEMPLATE_ET_H

#include <utility>
#include <RcppArmadillo.h>
#include "AbclassNet.h"
#include "AbclassGroupLasso.h"
#include "AbclassGroupSCAD.h"
#include "AbclassGroupMCP.h"
#include "utils.h"

namespace abclass {

    template <typename T>
    inline void et_group_lambda(T& obj,
                                const unsigned int nstage = 1)
    {
        const unsigned int p0 { obj.p0_ };
        const arma::mat x0 { obj.x_ };
        obj.set_group_weight();
        const arma::vec gw0 { obj.control_.group_weight_ };
        // initialize
        arma::mat active_x { obj.x_ };
        arma::uvec active_idx { arma::regspace<arma::uvec>(0, p0 - 1) };
        arma::mat active_beta;
        for (size_t i { 0 }; i < nstage; ++i) {
            // create pseudo-features
            arma::uvec perm_idx { arma::randperm(obj.n_obs_) };
            arma::mat x_perm { x0.rows(perm_idx) };
            x_perm = arma::join_rows(active_x, std::move(x_perm));
            obj.control_.group_weight_ = arma::join_cols(
                obj.control_.group_weight_, gw0);
            obj.set_data(x_perm, obj.y_);
            obj.permuted_ = p0;
            obj.fit();
            // reset lambda
            if (! obj.custom_lambda_) {
                obj.control_.lambda_.clear();
            }
            // update active x
            size_t p1_i { obj.p1_ - p0 };
            size_t p0_i { obj.p0_ - p0 };
            active_beta = obj.coef_.slice(
                obj.coef_.n_slices - 1).head_rows(p1_i);
            arma::vec l1_beta { arma::zeros(p1_i) };
            // get the indices of the selected predictors
            for (size_t row_i {0}; row_i < p1_i; ++row_i) {
                l1_beta[row_i] = l1_norm(active_beta.row(row_i));
            }
            arma::uvec idx0 { arma::find(l1_beta.tail_rows(p0_i) > 0) };
            active_x = active_x.cols(idx0);
            active_idx = active_idx.elem(arma::find(l1_beta > 0));
            obj.control_.group_weight_ = obj.control_.group_weight_.elem(idx0);
        }
        // update obj
        obj.set_data(x0, obj.y_);
        obj.coef_ = arma::cube(obj.p1_, obj.k_ - 1, 1, arma::fill::zeros);
        obj.coef_.slice(0).rows(active_idx) = active_beta.rows(active_idx);
        obj.permuted_ = 0;
    }

}

#endif
