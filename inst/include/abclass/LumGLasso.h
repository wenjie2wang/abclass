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

#ifndef ABCLASS_LUM_GLASSO_H
#define ABCLASS_LUM_GLASSO_H

#include <RcppArmadillo.h>
#include "AbclassGroupLasso.h"
#include "utils.h"

namespace abclass
{
    // define class for inputs and outputs
    class LumGLasso : public AbclassGroupLasso
    {
    private:
        // cache
        double lum_cp1_;        // c + 1
        double lum_c_cp1_;      // c / (c + 1)
        double lum_cma_;        // c - a
        double lum_ap1_;        // a + 1
        double lum_a_ap1_;       // a ^ (a + 1)

    protected:

        double lum_c_ = 0.0;    // c
        double lum_a_ = 1.0;    // a

        // set CMD lowerbound
        inline void set_gmd_lowerbound() override
        {
            double tmp { lum_ap1_ / lum_a_ * lum_cp1_ };
            arma::mat sqx { arma::square(x_) };
            sqx.each_col() %= obs_weight_;
            gmd_lowerbound_ = tmp * arma::sum(sqx, 0) / dn_obs_;
        }

        // objective function without regularization
        inline double objective0(const arma::vec& inner) const override
        {
            arma::vec tmp { arma::zeros(inner.n_elem) };
            for (size_t i {0}; i < inner.n_elem; ++i) {
                if (inner[i] < lum_c_cp1_) {
                    tmp[i] = 1.0 - inner[i];
                } else {
                    tmp[i] = std::pow(lum_a_ / (lum_cp1_ * inner[i] - lum_cma_),
                                      lum_a_) / lum_cp1_;
                }
            }
            return arma::mean(obs_weight_ % tmp);
        }

        // the first derivative of the loss function
        inline arma::vec loss_derivative(const arma::vec& u) const override
        {
            arma::vec out { - arma::ones(u.n_elem) };
            for (size_t i {0}; i < u.n_elem; ++i) {
                if (u[i] > lum_c_cp1_) {
                    out[i] = - lum_a_ap1_ /
                        std::pow(lum_cp1_ * u[i] - lum_cma_, lum_ap1_);
                }
            }
            return out;
        }

    public:

        // inherit constructors
        using AbclassGroupLasso::AbclassGroupLasso;

        //! @param x The design matrix without an intercept term.
        //! @param y The category index vector.
        LumGLasso(const arma::mat& x,
                  const arma::uvec& y,
                  const double lum_a = 1.0,
                  const double lum_c = 0.0,
                  const bool intercept = true,
                  const bool standardize = true,
                  const arma::vec& weight = arma::vec()) :
            AbclassGroupLasso(x, y, intercept, standardize, weight)
        {
            set_lum_parameters(lum_a, lum_c);
        }

        LumGLasso* set_lum_parameters(const double lum_a,
                                      const double lum_c)
        {
            if (is_le(lum_a, 0.0)) {
                throw std::range_error("The LUM 'a' must be positive.");
            }
            lum_a_ = lum_a;
            lum_ap1_ = lum_a_ + 1.0;
            lum_a_ap1_ = std::pow(lum_a_, lum_ap1_);
            if (is_lt(lum_c, 0.0)) {
                throw std::range_error("The LUM 'c' cannot be negative.");
            }
            lum_c_ = lum_c;
            lum_cp1_ = lum_c + 1.0;
            lum_c_cp1_ = lum_c_ / lum_cp1_;
            lum_cma_ = lum_c_ - lum_a_;
            return this;
        }

    };                          // end of class

}

#endif
