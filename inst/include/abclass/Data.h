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

#ifndef ABCLASS_DATA_H
#define ABCLASS_DATA_H

#include <RcppArmadillo.h>
#include <stdexcept>

#include "Control.h"
#include "utils.h"

namespace abclass
{

    class Simplex
    {
    public:
        unsigned int km1_{0}; // k - 1
        unsigned int k_{0};   // dimensions
        double dk_{0.0};      // double(k_)

        // k vertex column vectors in R^(k-1) => (k-1) by k
        arma::mat vertex_; // unique vertex: (k-1) by k

        // default constructor
        Simplex() {}
        Simplex(const unsigned int k) { set_k(k); }

        inline void set_k(const unsigned int k)
        {
            if (k < 2) {
                throw std::invalid_argument("k must be an integer > 1.");
            }
            k_ = k;
            km1_ = k - 1;
            dk_ = static_cast<double>(k);
            double dkm1{dk_ - 1.0};
            vertex_ = arma::zeros(km1_, k_);
            const arma::vec tmp{arma::ones<arma::vec>(km1_)};
            vertex_.col(0) = std::pow(dkm1, -0.5) * tmp;
            for (size_t j{1}; j < k_; ++j) {
                vertex_.col(j) =
                    -(1.0 + std::sqrt(k_)) / std::pow(dkm1, 1.5) * tmp;
                vertex_(j - 1, j) += std::sqrt(k_ / dkm1);
            }
        }
    };

    // a container for all the elements for computing loss
    template <typename T_x = arma::mat> class Data : public Simplex
    {
    protected:
        // for angle-based classification
        inline void set_ex_vertex(const arma::uvec& y)
        {
            t_vertex_ = vertex_.t();
            ex_vertex_ = arma::mat(y.n_elem, km1_);
            for (size_t i{0}; i < y.n_elem; ++i) {
                ex_vertex_.row(i) = t_vertex_.row(y[i]);
            }
        }

        // more general (e.g., for outcome-weighted learning)
        inline void set_ex_vertex(const arma::uvec& y, const arma::vec& factor)
        {
            t_vertex_ = vertex_.t();
            ex_vertex_ = arma::mat(y.n_elem, km1_);
            for (size_t i{0}; i < y.n_elem; ++i) {
                ex_vertex_.row(i) = t_vertex_.row(y[i]) * factor(i);
            }
        }

    public:
        arma::mat t_vertex_;  // transpose of vertex_: (K, K - 1)
        arma::mat ex_vertex_; // expanded vertex for y: n by (K - 1)
        arma::uvec y_;        // {0,1,...,k-1}

        unsigned int n_obs_; // number of observations
        double div_n_obs_;   // 1.0 / n_obs_
        unsigned int p0_;    // number of predictors without intercept
        unsigned int p1_;    // number of predictors (with intercept)
        unsigned int inter_; // integer version of intercept_

        T_x x_;                 // (standardized) x_: n by p (without intercept)
        arma::rowvec x_center_; // the column center of x_
        arma::rowvec x_scale_;  // the column scale of x_
        arma::uvec x_skip_;     // index of const x_

        arma::vec obs_weights_{arma::vec()};
        bool custom_obs_weight_{false};
        arma::vec offset_{arma::vec()};
        bool has_offset_{false};
        bool intercept_{true};
        bool standardize_{true};

        // tuning by ET-Lasso
        unsigned int et_npermuted_{0}; // number of permuted predictors

        // constructors
        Data() {}

        Data(const unsigned int k, const bool intercept = true,
             const bool standardize = true)
            : Simplex{k}
        {
            set_intercept(intercept);
            set_standardize(standardize);
        }

        Data(const T_x& x, const arma::uvec& y = arma::uvec(),
             const arma::vec& obs_weights = arma::vec(),
             const arma::vec& offset = arma::vec())
        {
            if (y.empty()) {
                set_x(x);
            } else {
                set_data(x, y);
            }
            set_obs_weights(obs_weights);
            set_offset(offset);
        }

        // setters
        inline void set_y(const arma::uvec& y)
        {
            // assume y in {0, ..., k-1}
            // assume binary classification if y takes zero only
            // check if a valid k is set; honor the set k
            if (k_ < 2) {
                set_k(std::max(2U, y.max() + 1));
            }
            y_ = y;
            set_ex_vertex(y);
            // make sure obs_weights is set
            if (obs_weights_.empty()) {
                set_obs_weights();
            };
        }

        inline void set_x(const T_x& x)
        {
            x_ = x;
            n_obs_ = x.n_rows;
            div_n_obs_ = 1.0 / static_cast<double>(n_obs_);
            inter_ = static_cast<unsigned int>(intercept_);
            p0_ = x_.n_cols;
            p1_ = p0_ + inter_;
            if (standardize_) {
                if (intercept_) {
                    x_center_ = arma::mean(x_);
                } else {
                    x_center_ = arma::zeros<arma::rowvec>(p0_);
                }
                x_scale_ = col_sd(x_);
                for (size_t j{0}; j < p0_; ++j) {
                    if (x_scale_(j) > 0) {
                        x_.col(j) = (x_.col(j) - x_center_(j)) / x_scale_(j);
                    } else {
                        x_.col(j).zeros();
                        // make scale(j) nonzero for rescaling
                        x_scale_(j) = -1.0;
                    }
                }
            } else {
                x_scale_ = col_sd(x_);
            }
            x_skip_ = arma::find(x_scale_ <= 0.0);
        }

        // common method to set data
        inline void set_data(const T_x& x, const arma::uvec& y)
        {
            set_y(y);
            set_x(x);
        }

        // for outcome-weighted learning
        // need set data first
        inline void set_owl_reward(const arma::vec& reward = arma::vec())
        {
            if (reward.empty()) {
                return;         // do nothing
            }
            if (reward.n_elem != n_obs_) {
                throw std::invalid_argument(
                    "Inconsistent length of internal 'reward'.");
            }
            for (size_t i{0}; i < n_obs_; ++i) {
                ex_vertex_.row(i) *= sign(reward(i));
            }
        }

        inline void set_intercept(const bool intercept = true)
        {
            intercept_ = intercept;
        }

        inline void set_standardize(const bool standardize = true)
        {
            standardize_ = standardize;
        }

        inline void set_obs_weights(const arma::vec& weights = arma::vec())
        {
            if (weights.empty()) {
                obs_weights_ = arma::ones(n_obs_);
                custom_obs_weight_ = false;
                return;
            } else if (weights.n_elem == n_obs_) {
                obs_weights_ = weights / (arma::accu(weights) * div_n_obs_);
                custom_obs_weight_ = true;
                return;
            }
            throw std::invalid_argument("Incorrect length of "
                                        "observational weights.");
        }

        inline void set_offset(const arma::mat& offset = arma::mat())
        {
            if (offset.empty() || offset.is_zero()) {
                offset_ = arma::mat();
                has_offset_ = false;
                return;
            }
            if (offset.n_rows == n_obs_ && offset.n_cols == km1_) {
                offset_ = offset;
                has_offset_ = true;
                return;
            }
            throw std::invalid_argument("Inconsistent length of offsets.");
        }

    };

   // cache for iterative estimation procedure
    class IterCache
    {
    public:
        arma::vec iter_inner_;  // n x 1
        arma::mat iter_pred_f_; // n x (K - 1)
        arma::vec iter_vk_xg_;  // n x 1
        arma::mat iter_v_xg_;   // n x (K - 1)

        IterCache() {}

        // reset cache
        inline void reset_cache()
        {
            iter_inner_.reset();
            iter_pred_f_.reset();
            iter_vk_xg_.reset();
            iter_v_xg_.reset();
        }
    };

    // common output data container for Abclass
    template <typename T_x = arma::mat>
    class Output
    {
    public:
        // estimates
        arma::cube coef_; // p1_ x km1_ for linear learning in each slice

        // loss/penalty/objective functions along the solution path
        arma::vec loss_;
        arma::vec penalty_;
        arma::vec objective_;

        unsigned int n_iter_; // number of iteration

        // tuning by cross-validation
        arma::mat cv_accuracy_;
        arma::vec cv_accuracy_mean_;
        arma::vec cv_accuracy_sd_;

        // tuning by ET-Lasso
        arma::uvec et_vs_;             // indices of selected predictors

        // one time value for one stage
        // the smallest lambda before selection of any random predictors
        double et_l1_lambda0_; // the last lambda before the cutoff
        double et_l1_lambda1_; // the cutoff point

        // to save values from all the stages for output
        arma::vec et_l1_lambda0_vec_;
        arma::vec et_l1_lambda1_vec_;

        // regularization
        // the "big" enough lambda => zero coef unless alpha = 0
        double l1_lambda_max_;
        double lambda_max_;

        // control
        Control control_;

        // pointer to the training data
        const Data<T_x>* data_;

        // constructor
        Output() {}
    };

} // namespace abclass

#endif
