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

#include <stdexcept>
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
    abclass::Control ctrl { abclass_control(control) };
    switch (loss_id) {
        case 1: {               // logistic
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::LogisticNet<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::LogisticSCAD<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::LogisticMCP<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::LogisticGroupLasso<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::LogisticGroupSCAD<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::LogisticGroupMCP<T> object {x, y, ctrl};
                    return template_fit(object);
                }
            }
        }
        case 2: {               // boost
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::BoostNet<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::BoostSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::BoostMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::BoostGroupLasso<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::BoostGroupSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::BoostGroupMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
            }
        }
        case 3: {               // hinge-boost
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::HingeBoostNet<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::HingeBoostSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::HingeBoostMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::HingeBoostGroupLasso<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::HingeBoostGroupSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::HingeBoostGroupMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
            }
        }
        case 4: {               // lum
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::LumNet<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::LumSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::LumMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::LumGroupLasso<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::LumGroupSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::LumGroupMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
            }
        }
        default:
            break;
    }
    throw std::range_error("Invalid choice of loss or penalty.");
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
