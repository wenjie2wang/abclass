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

#ifndef ABCLASS_SIMPLEX_H
#define ABCLASS_SIMPLEX_H

#include <RcppArmadillo.h>
#include <stdexcept>

namespace abclass
{

    class Simplex
    {
    public:
        unsigned int km1_;      // k - 1
        unsigned int k_;        // dimensions
        double dk_;             // double(k_)

        // k vertex column vectors in R^(k-1) => (k-1) by k
        arma::mat vertex_;      // unique vertex: (k-1) by k

        // default constructor
        Simplex(const unsigned int k = 2)
        {
            update_k(k);
        }

        inline void update_k(const unsigned int k)
        {
            if (k < 2) {
                throw std::range_error("k must be an integer > 1.");
            }
            k_ = k;
            km1_ = k - 1;
            dk_ = static_cast<double>(k);
            double dkm1 { dk_ - 1.0 };
            vertex_ = arma::zeros(km1_, k_);
            const arma::vec tmp { arma::ones<arma::vec>(km1_) };
            vertex_.col(0) = std::pow(dkm1, - 0.5) * tmp;
            for (size_t j {1}; j < k_; ++j) {
                vertex_.col(j) = - (1.0 + std::sqrt(k_)) /
                    std::pow(dkm1, 1.5) * tmp;
                vertex_(j - 1, j) += std::sqrt(k_ / dkm1);
            }
        }

    };

    // another simplex class that contains all the elements for computing loss
    template<typename T_x>
    class Simplex2 : public Simplex
    {
    public:
        arma::mat t_vertex_;    // transpose of vertex_: (K, K - 1)
        arma::mat ex_vertex_;   // expanded vertex for y: n by (K - 1)
        arma::uvec y_;          // {0,1,...,k-1}

        unsigned int n_obs_;    // number of observations
        double div_n_obs_;      // 1.0 / n_obs_
        unsigned int p0_;       // number of predictors without intercept
        unsigned int p1_;       // number of predictors (with intercept)
        unsigned int inter_;    // integer version of intercept_

        T_x x_;                 // (standardized) x_: n by p (without intercept)
        arma::rowvec x_center_; // the column center of x_
        arma::rowvec x_scale_;  // the column scale of x_
        arma::uvec x_skip_;     // index of const x_

        // cache for iterative estimation procedure
        arma::vec iter_inner_;  // n x 1
        arma::mat iter_pred_f_; // n x (K - 1)
        arma::vec iter_vk_xg_;  // n x 1
        arma::mat iter_v_xg_;   // n x (K - 1)

        Simplex2(const unsigned int k = 2) :
            Simplex { k }
        {
            t_vertex_ = vertex_.t();
        }

        // for angle-based classification
        inline void set_ex_vertex(const arma::uvec& y)
        {
            ex_vertex_ = arma::mat(y.n_elem, km1_);
            for (size_t i {0}; i < y.n_elem; ++i) {
                ex_vertex_.row(i) = t_vertex_.row(y[i]);
            }
        }
        // more general (e.g., for outcome-weighted learning)
        inline void set_ex_vertex(const arma::uvec& y, const arma::vec& factor)
        {
            ex_vertex_ = arma::mat(y.n_elem, km1_);
            for (size_t i {0}; i < y.n_elem; ++i) {
                ex_vertex_.row(i) = t_vertex_.row(y[i]) * factor(i);
            }
        }

        // reset cache
        inline void reset_cache()
        {
            iter_inner_.reset();
            iter_pred_f_.reset();
            iter_vk_xg_.reset();
            iter_v_xg_.reset();
        }

    };

}

#endif
