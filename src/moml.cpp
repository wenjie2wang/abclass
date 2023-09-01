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
    const size_t method_id { penalty_id * 100 + loss_id };
    abclass::Control ctrl { abclass_control(control) };
    switch (method_id) {
        case 101: {
            abclass::MomlNet<abclass::Logistic, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            return template_fit(object);
        }
        case 102: {
            abclass::MomlNet<abclass::Boost, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_inner_min(control["boost_umin"]);
            return template_fit(object);
        }
        case 103: {
            abclass::MomlNet<abclass::HingeBoost, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_c(control["lum_c"]);
            return template_fit(object);
        }
        case 104: {
            abclass::MomlNet<abclass::Lum, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_fit(object);
        }
        case 201: {
            abclass::MomlGroupLasso<abclass::Logistic, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            return template_fit(object);
        }
        case 202: {
            abclass::MomlGroupLasso<abclass::Boost, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_inner_min(control["boost_umin"]);
            return template_fit(object);
        }
        case 203: {
            abclass::MomlGroupLasso<abclass::HingeBoost, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_c(control["lum_c"]);
            return template_fit(object);
        }
        case 204: {
            abclass::MomlGroupLasso<abclass::Lum, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_fit(object);
        }
        case 301: {
            abclass::MomlGroupSCAD<abclass::Logistic, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            return template_fit(object);
        }
        case 302: {
            abclass::MomlGroupSCAD<abclass::Boost, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_inner_min(control["boost_umin"]);
            return template_fit(object);
        }
        case 303: {
            abclass::MomlGroupSCAD<abclass::HingeBoost, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_c(control["lum_c"]);
            return template_fit(object);
        }
        case 304: {
            abclass::MomlGroupSCAD<abclass::Lum, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_fit(object);
        }
        case 401: {
            abclass::MomlGroupMCP<abclass::Logistic, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            return template_fit(object);
        }
        case 402: {
            abclass::MomlGroupMCP<abclass::Boost, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_inner_min(control["boost_umin"]);
            return template_fit(object);
        }
        case 403: {
            abclass::MomlGroupMCP<abclass::HingeBoost, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_c(control["lum_c"]);
            return template_fit(object);
        }
        case 404: {
            abclass::MomlGroupMCP<abclass::Lum, T> object {
                x, treatment, reward, propensity_score, ctrl
            };
            object.loss_.set_ac(control["lum_a"], control["lum_c"]);
            return template_fit(object);
        }
        default:
            break;
    }
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
