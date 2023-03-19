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

#include <RcppArmadillo.h>
#include <abclass.h>
#include "template_helpers.h"

// template interface
template <typename T>
Rcpp::List template_abclass_fit(
    const T& x,
    const arma::uvec& y,
    const Rcpp::List& control
)
{
    const size_t loss_id { control["loss_id"] };
    const size_t penalty_id { control["penalty_id"] };
    const size_t method_id { penalty_id * 100 + loss_id };
    abclass::Control ctrl { conv_control(control) };
    switch (method_id) {
        case 101: {
            abclass::LogisticNet<T> object {x, y, ctrl};
            return template_fit(object);
        }
        case 102: {
            abclass::BoostNet<T> object {x, y, ctrl};
            object.loss_.set_inner_min(control["boost_umin"]);
            return template_fit(object);
        }
        case 103: {
            abclass::HingeBoostNet<T> object {x, y, ctrl};
            object.loss_.set_c(control["lum_c"]);
            return template_fit(object);
        }
        case 104: {
            abclass::LumNet<T> object {x, y, ctrl};
            object.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_fit(object);
        }
        case 201: {
            abclass::LogisticGroupLasso<T> object {x, y, ctrl};
            return template_fit(object);
        }
        case 202: {
            abclass::BoostGroupLasso<T> object {x, y, ctrl};
            object.loss_.set_inner_min(control["boost_umin"]);
            return template_fit(object);
        }
        case 203: {
            abclass::HingeBoostGroupLasso<T> object {x, y, ctrl};
            object.loss_.set_c(control["lum_c"]);
            return template_fit(object);
        }
        case 204: {
            abclass::LumGroupLasso<T> object {x, y, ctrl};
            object.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_fit(object);
        }
        case 301: {
            abclass::LogisticGroupSCAD<T> object {x, y, ctrl};
            return template_fit(object);
        }
        case 302: {
            abclass::BoostGroupSCAD<T> object {x, y, ctrl};
            object.loss_.set_inner_min(control["boost_umin"]);
            return template_fit(object);
        }
        case 303: {
            abclass::HingeBoostGroupSCAD<T> object {x, y, ctrl};
            object.loss_.set_c(control["lum_c"]);
            return template_fit(object);
        }
        case 304: {
            abclass::LumGroupSCAD<T> object {x, y, ctrl};
            object.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_fit(object);
        }
        case 401: {
            abclass::LogisticGroupMCP<T> object {x, y, ctrl};
            return template_fit(object);
        }
        case 402: {
            abclass::BoostGroupMCP<T> object {x, y, ctrl};
            object.loss_.set_inner_min(control["boost_umin"]);
            return template_fit(object);
        }
        case 403: {
            abclass::HingeBoostGroupMCP<T> object {x, y, ctrl};
            object.loss_.set_c(control["lum_c"]);
            return template_fit(object);
        }
        case 404: {
            abclass::LumGroupMCP<T> object {x, y, ctrl};
            object.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_fit(object);
        }
        default:
            break;
    }
    return Rcpp::List();
}

// [[Rcpp::export]]
Rcpp::List rcpp_abclass_fit(
    const arma::mat& x,
    const arma::uvec& y,
    const Rcpp::List& control
)
{
    return template_abclass_fit<arma::mat>(x, y, control);
}

// [[Rcpp::export]]
Rcpp::List rcpp_abclass_fit_sp(
    const arma::sp_mat& x,
    const arma::uvec& y,
    const Rcpp::List& control
)
{
    return template_abclass_fit<arma::sp_mat>(x, y, control);
}
