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

#include <RcppArmadillo.h>
#include <abclass.h>
#include "template_helpers.h"

// template interface
template <typename T>
Rcpp::List template_moml_fit(
    const T& x,
    const arma::uvec& treatment,
    const arma::vec& reward,
    const arma::vec& propensity_score,
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
                    abclass::Moml<abclass::LogisticNet<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::Moml<abclass::LogisticSCAD<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::Moml<abclass::LogisticMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::Moml<abclass::LogisticGroupLasso<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::Moml<abclass::LogisticGroupSCAD<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::Moml<abclass::LogisticGroupMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::Moml<abclass::LogisticCompMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::Moml<abclass::LogisticGEL<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
                case 9: {       // mellowmax
                    abclass::Moml<abclass::LogisticMellowmax<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
                case 10: {      // mellowmax mcp
                    abclass::Moml<abclass::LogisticMellowMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    return template_fit(object);
                }
            }
        }
        case 2: {               // boost
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::Moml<abclass::BoostNet<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::Moml<abclass::BoostSCAD<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::Moml<abclass::BoostMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::Moml<abclass::BoostGroupLasso<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::Moml<abclass::BoostGroupSCAD<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::Moml<abclass::BoostGroupMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::Moml<abclass::BoostCompMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::Moml<abclass::BoostGEL<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 9: {       // mellowmax
                    abclass::Moml<abclass::BoostMellowmax<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
                case 10: {       // mellowmax mcp
                    abclass::Moml<abclass::BoostMellowMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_inner_min(control["boost_umin"]);
                    return template_fit(object);
                }
            }
        }
        case 3: {               // hinge-boost
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::Moml<abclass::HingeBoostNet<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::Moml<abclass::HingeBoostSCAD<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::Moml<abclass::HingeBoostMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::Moml<abclass::HingeBoostGroupLasso<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::Moml<abclass::HingeBoostGroupSCAD<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::Moml<abclass::HingeBoostGroupMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::Moml<abclass::HingeBoostCompMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::Moml<abclass::HingeBoostGEL<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 9: {       // mellowmax
                    abclass::Moml<abclass::HingeBoostMellowmax<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
                case 10: {       // mellowmax mcp
                    abclass::Moml<abclass::HingeBoostMellowMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_c(control["lum_c"]);
                    return template_fit(object);
                }
            }
        }
        case 4: {               // lum
            switch (penalty_id) {
                case 1: {       // lasso
                    abclass::Moml<abclass::LumNet<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 2: {       // scad
                    abclass::Moml<abclass::LumSCAD<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 3: {       // mcp
                    abclass::Moml<abclass::LumMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 4: {       // group lasso
                    abclass::Moml<abclass::LumGroupLasso<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 5: {       // group scad
                    abclass::Moml<abclass::LumGroupSCAD<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 6: {       // group mcp
                    abclass::Moml<abclass::LumGroupMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 7: {       // composite mcp
                    abclass::Moml<abclass::LumCompMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 8: {       // gel
                    abclass::Moml<abclass::LumGEL<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 9: {       // mellowmax
                    abclass::Moml<abclass::LumMellowmax<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
                    object.loss_fun_.set_ac(control["lum_a"], control["lum_c"]);
                    return template_fit(object);
                }
                case 10: {       // mellowmax mcp
                    abclass::Moml<abclass::LumMellowMCP<T>, T> object {
                        x, treatment, reward, propensity_score, ctrl
                    };
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
Rcpp::List rcpp_moml_fit(
    const arma::mat& x,
    const arma::uvec& treatment,
    const arma::vec& reward,
    const arma::vec& propensity_score,
    const Rcpp::List& control
    )
{
    return template_moml_fit<arma::mat>(x, treatment, reward,
                                        propensity_score, control);
}

// [[Rcpp::export]]
Rcpp::List rcpp_moml_fit_sp(
    const arma::sp_mat& x,
    const arma::uvec& treatment,
    const arma::vec& reward,
    const arma::vec& propensity_score,
    const Rcpp::List& control
    )
{
    return template_moml_fit<arma::sp_mat>(x, treatment, reward,
                                           propensity_score, control);
}
