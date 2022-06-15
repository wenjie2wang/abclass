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
    inline void et_lambda(T& obj)
    {
        // record some original data
        const unsigned int p0 { obj.p0_ };
        const unsigned int inter { obj.p1_ - obj.p0_ };
        const auto x0 { obj.x_ };
        obj.set_group_weight();
        const arma::vec gw0 { obj.control_.group_weight_ };
        // initialize
        // (0, 1, ...p0 - 1), assuming p0 > 0
        obj.et_vs_ = arma::regspace<arma::uvec>(0, p0 - 1);
        arma::mat active_beta;
        arma::uvec active_idx0;
        for (size_t i { 0 }; i < obj.control_.et_nstages_; ++i) {
            // create pseudo-features
            const arma::uvec perm_idx { arma::randperm(obj.n_obs_) };
            auto x_perm { subset_rows(x0, perm_idx) };
            x_perm = arma::join_rows(x0.cols(obj.et_vs_), std::move(x_perm));
            obj.control_.group_weight_ = arma::join_cols(
                obj.control_.group_weight_.elem(obj.et_vs_), gw0);
            obj.set_data(x_perm, obj.y_);
            obj.et_npermuted_ = p0;
            obj.fit();
            // reset lambda if it was internally set
            if (! obj.custom_lambda_) {
                obj.control_.lambda_.clear();
            }
            // update active x
            const unsigned int p1_i { obj.p1_ - p0 };
            const unsigned int p0_i { obj.p0_ - p0 };
            active_beta = obj.coef_.slice(
                obj.coef_.n_slices - 1).head_rows(p1_i);
            arma::vec l1_beta { arma::zeros(p0_i) };
            // get the indices of the selected predictors
            for (size_t j { 0 }; j < p0_i; ++j) {
                l1_beta[j] = l1_norm(active_beta.row(inter + j));
            }
            active_idx0 = arma::find(l1_beta > 0);
            obj.et_vs_ = obj.et_vs_.elem(active_idx0);
            // verbose
            if (obj.control_.verbose_ > 0) {
                Rcpp::Rcout << "[ET] (stage "
                            << i + 1
                            << ") Number of active predictors: "
                            << obj.et_vs_.n_elem
                            << "\n";
            }
        }
        // update obj
        obj.set_data(std::move(x0), obj.y_);
        obj.set_group_weight(std::move(gw0));
        obj.coef_ = arma::cube(obj.p1_, obj.k_ - 1, 1, arma::fill::zeros);
        if (obj.control_.intercept_) {
            obj.coef_.slice(0).rows(obj.et_vs_ + 1) =
                active_beta.rows(active_idx0 + 1);
            // intercept
            obj.coef_.slice(0).row(0) = active_beta.row(0);
        } else {
            obj.coef_.slice(0).rows(obj.et_vs_) = active_beta.rows(active_idx0);
        }
        obj.et_npermuted_ = 0;
    }

}

#endif
