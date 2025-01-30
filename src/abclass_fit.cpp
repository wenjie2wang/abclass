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
                    abclass::LogitNet<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::LogitSCAD<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::LogitMCP<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::LogitGLasso<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::LogitGSCAD<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::LogitGMCP<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::LogitCMCP<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::LogitGEL<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 9: {       // mellowmax L1
                    abclass::LogitML1<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 10: {      // mellowmax mcp
                    abclass::LogitMMCP<T> object {x, y, ctrl};
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
                    abclass::BoostGLasso<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::BoostGSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::BoostGMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::BoostCMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::BoostGEL<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 9: {       // mellowmax L1
                    abclass::BoostML1<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 10: {       // mellowmax mcp
                    abclass::BoostMMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
            }
        }
        case 3: {               // hinge.boost
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::HBoostNet<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::HBoostSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::HBoostMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::HBoostGLasso<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::HBoostGSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::HBoostGMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::HBoostCMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::HBoostGEL<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 9: {       // mellowmax L1
                    abclass::HBoostML1<T> object {x, y, ctrl};
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 10: {       // mellowmax mcp
                    abclass::HBoostMMCP<T> object {x, y, ctrl};
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
                    abclass::LumGLasso<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::LumGSCAD<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::LumGMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::LumCMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::LumGEL<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 9: {       // mellowmax L1
                    abclass::LumML1<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 10: {       // mellowmax mcp
                    abclass::LumMMCP<T> object {x, y, ctrl};
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
            }
        }
        case 5: {               // mlogit
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::MlogitNet<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::MlogitSCAD<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::MlogitMCP<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::MlogitGLasso<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::MlogitGSCAD<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::MlogitGMCP<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::MlogitCMCP<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::MlogitGEL<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 9: {       // mellowmax L1
                    abclass::MlogitML1<T> object {x, y, ctrl};
                    return template_fit(object);
                }
                case 10: {      // mellowmax mcp
                    abclass::MlogitMMCP<T> object {x, y, ctrl};
                    return template_fit(object);
                }
            }
        }
        case 6: {               // LikeLogistic
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::LeLogitNet<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::LeLogitSCAD<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::LeLogitMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::LeLogitGLasso<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::LeLogitGSCAD<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::LeLogitGMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::LeLogitCMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::LeLogitGEL<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 9: {       // mellowmax L1
                    abclass::LeLogitML1<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 10: {      // mellowmax mcp
                    abclass::LeLogitMMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
            }
        }
        case 7: {               // LikeBoost
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::LeBoostNet<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::LeBoostSCAD<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::LeBoostMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::LeBoostGLasso<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::LeBoostGSCAD<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::LeBoostGMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::LeBoostCMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::LeBoostGEL<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 9: {       // mellowmax L1
                    abclass::LeBoostML1<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 10: {      // mellowmax mcp
                    abclass::LeBoostMMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
            }
        }
        case 8: {               // LikeHingeBoost
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::LeHBoostNet<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::LeHBoostSCAD<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::LeHBoostMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::LeHBoostGLasso<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::LeHBoostGSCAD<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::LeHBoostGMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::LeHBoostCMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::LeHBoostGEL<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 9: {       // mellowmax L1
                    abclass::LeHBoostML1<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 10: {      // mellowmax mcp
                    abclass::LeHBoostMMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
            }
        }
        case 9: {               // LikeLum
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::LeLumNet<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::LeLumSCAD<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::LeLumMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::LeLumGLasso<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::LeLumGSCAD<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::LeLumGMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::LeLumCMCP<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::LeLumGEL<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 9: {       // mellowmax L1
                    abclass::LeLumML1<T> object {
                        x, y, ctrl
                    };
                    return template_fit(object);
                }
                case 10: {      // mellowmax mcp
                    abclass::LeLumMMCP<T> object {
                        x, y, ctrl
                    };
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
