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

#ifndef ABCLASS_ABCLASS_GROUP_H
#define ABCLASS_ABCLASS_GROUP_H

#include <RcppArmadillo.h>
#include "Abclass.h"
#include "Control.h"

namespace abclass
{
    // base class for the angle-based large margin classifiers
    // with group-wise regularization

    // T_x is intended to be arma::mat or arma::sp_mat
    template <typename T_loss, typename T_x>
    class AbclassGroup : public Abclass<T_loss, T_x>
    {
    protected:
        using Abclass<T_loss, T_x>::dn_obs_;
        using Abclass<T_loss, T_x>::km1_;
        using Abclass<T_loss, T_x>::loss_derivative;

        // define gradient function for j-th predictor
        inline arma::rowvec mm_gradient(const arma::vec& inner,
                                        const unsigned int j) const
        {
            arma::vec inner_grad { loss_derivative(inner) };
            arma::vec tmp_vec {
                x_.col(j) % control_.obs_weight_ % inner_grad
            };
            arma::rowvec out { tmp_vec.t() * ex_vertex_ };
            return out / dn_obs_;
        }
        // for intercept
        inline arma::rowvec mm_gradient0(const arma::vec& inner) const
        {
            arma::vec inner_grad { loss_derivative(inner) };
            arma::vec tmp_vec { control_.obs_weight_ % inner_grad };
            arma::rowvec out { tmp_vec.t() * ex_vertex_ };
            return out / dn_obs_;
        }

        // gradient matrix for beta
        inline arma::mat gradient(const arma::vec& inner) const
        {
            arma::mat out { arma::zeros(p0_, km1_) };
            arma::vec inner_grad { loss_derivative(inner) };
            for (size_t j {0}; j < p0_; ++j) {
                arma::vec tmp_vec {
                    x_.col(j) % control_.obs_weight_ % inner_grad
                };
                arma::rowvec tmp { tmp_vec.t() * ex_vertex_ };
                out.row(j) = tmp;
            }
            return out / dn_obs_;
        }

    public:
        // inherit constructors
        using Abclass<T_loss, T_x>::Abclass;
        using Abclass<T_loss, T_x>::control_;
        using Abclass<T_loss, T_x>::n_obs_;
        using Abclass<T_loss, T_x>::p0_;
        using Abclass<T_loss, T_x>::x_;
        using Abclass<T_loss, T_x>::y_;
        using Abclass<T_loss, T_x>::ex_vertex_;
        using Abclass<T_loss, T_x>::et_npermuted_;
        using Abclass<T_loss, T_x>::coef_;

        // regularization
        // the "big" enough lambda => zero coef
        double lambda_max_;
        // did user specified a customized lambda sequence?
        bool custom_lambda_ = false;

        // for a sequence of lambda's
        virtual void fit() = 0;

    };

}

#endif /* ABCLASS_ABCLASS_GROUP_H */
