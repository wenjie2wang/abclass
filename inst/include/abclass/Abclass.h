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

#ifndef ABCLASS_ABCLASS_H
#define ABCLASS_ABCLASS_H

#include <RcppArmadillo.h>
#include "Control.h"
#include "Simplex.h"

namespace abclass
{
    // base class for the angle-based large margin classifiers
    // T_x is intended to be arma::mat or arma::sp_mat
    // T_loss should be one of the loss function classes
    template <typename T_loss, typename T_x = arma::mat>
    class Abclass
    {
    protected:

        // cache variables
        double dn_obs_;              // double version of n_obs_
        unsigned int km1_;           // k - 1
        unsigned int inter_;         // integer version of intercept_

        // for the CMD/GMD algorithm
        double mm_lowerbound0_;
        arma::rowvec mm_lowerbound_;

        // prepare the vertex matrix
        inline void set_vertex_matrix(const unsigned int k)
        {
            Simplex sim { k };
            vertex_ = sim.get_vertex();
        }
        inline void set_ex_vertex_matrix()
        {
            ex_vertex_ = arma::mat(n_obs_, km1_);
            for (size_t i {0}; i < n_obs_; ++i) {
                ex_vertex_.row(i) = vertex_.row(y_[i]);
            }
        }

        inline arma::vec get_vertex_y(const unsigned int j) const
        {
            // j in {0, 1, ..., k - 2}
            // arma::vec vj { vertex_.col(j) };
            // return vj.elem(y_);
            return ex_vertex_.col(j);
        }

        // transfer coef for standardized data to coef for non-standardized data
        inline arma::mat rescale_coef(const arma::mat& beta) const
        {
            arma::mat out { beta };
            if (control_.standardize_) {
                if (control_.intercept_) {
                    // for each columns
                    for (size_t k { 0 }; k < km1_; ++k) {
                        arma::vec coef_k { beta.col(k) };
                        out(0, k) = beta(0, k) -
                            arma::as_scalar((x_center_ / x_scale_) *
                                            coef_k.tail_rows(p0_));
                        for (size_t l { 1 }; l < p1_; ++l) {
                            out(l, k) = coef_k(l) / x_scale_(l - 1);
                        }
                    }
                } else {
                    for (size_t k { 0 }; k < km1_; ++k) {
                        for (size_t l { 0 }; l < p0_; ++l) {
                            out(l, k) /= x_scale_(l);
                        }
                    }
                }
            }
            return out;
        }

        // loss function
        inline double objective0(const arma::vec& inner) const
        {
            return loss_.loss(inner, control_.obs_weight_);
        }
        // the first derivative of the loss function
        inline arma::vec loss_derivative(const arma::vec& inner) const
        {
            return loss_.dloss(inner);
        }

        // MM lowerbound used in coordinate-descent algorithm
        inline void set_mm_lowerbound()
        {
            if (control_.intercept_) {
                mm_lowerbound0_ = loss_.mm_lowerbound(
                    dn_obs_, control_.obs_weight_);
            }
            mm_lowerbound_ = loss_.mm_lowerbound(x_, control_.obs_weight_);
        }

        inline arma::vec gen_group_weight(
            const arma::vec& group_weight = arma::vec()
            ) const
        {
            if (group_weight.n_elem < p0_) {
                arma::vec out { arma::ones(p0_) };
                if (group_weight.is_empty()) {
                    return out;
                }
            } else if (group_weight.n_elem == p0_) {
                if (arma::any(group_weight < 0.0)) {
                    throw std::range_error(
                        "The 'group_weight' cannot be negative.");
                }
                return group_weight;
            }
            // else
            throw std::range_error("Incorrect length of the 'group_weight'.");
        }

    public:

        // from the data
        unsigned int n_obs_;    // number of observations
        unsigned int k_;        // number of categories
        unsigned int p0_;       // number of predictors without intercept
        unsigned int p1_;       // number of predictors (with intercept)
        T_x x_;                 // (standardized) x_: n by p (without intercept)
        arma::uvec y_;          // y vector ranging in {0, ..., k - 1}
        arma::mat vertex_;      // unique vertex: k by (k - 1)
        arma::mat ex_vertex_;   // expanded vertex for y_: n by (k - 1)
        arma::rowvec x_center_; // the column center of x_
        arma::rowvec x_scale_;  // the column scale of x_

        // parameters
        Control control_;       // control parameters
        T_loss loss_;           // loss funciton class

        // tuning by cross-validation
        arma::mat cv_accuracy_;
        arma::vec cv_accuracy_mean_;
        arma::vec cv_accuracy_sd_;

        // tuning by ET-Lasso
        unsigned int et_npermuted_ { 0 }; // number of permuted predictors
        arma::uvec et_vs_;                // indices of selected predictors

        // estimates
        arma::cube coef_;       // p1_ by km1_ for linear learning

        // loss along the solution path
        arma::vec loss_wo_penalty_;
        arma::vec penalty_;

        // default constructor
        Abclass() {}

        // for using prediction functions
        explicit Abclass(const unsigned k)
        {
            set_vertex_matrix(k);
            k_ = k;
        }

        // main constructor
        Abclass(const T_x& x,
                const arma::uvec& y,
                const Control& control = Control()) :
            control_ (control)
        {
            set_data(x, y);
            set_weight(control_.obs_weight_);
        }

        // setter
        inline Abclass* set_data(const T_x& x,
                                 const arma::uvec& y)
        {
            x_ = x;
            y_ = y;
            inter_ = static_cast<unsigned int>(control_.intercept_);
            km1_ = arma::max(y_); // assume y in {0, ..., k-1}
            k_ = km1_ + 1;
            n_obs_ = x_.n_rows;
            dn_obs_ = static_cast<double>(n_obs_);
            p0_ = x_.n_cols;
            p1_ = p0_ + inter_;
            set_vertex_matrix(k_);
            set_ex_vertex_matrix();
            if (control_.standardize_) {
                if (control_.intercept_) {
                    x_center_ = arma::mean(x_);
                } else {
                    x_center_ = arma::zeros<arma::rowvec>(x_.n_cols);
                }
                x_scale_ = col_sd(x_);
                for (size_t j {0}; j < p0_; ++j) {
                    if (x_scale_(j) > 0) {
                        x_.col(j) = (x_.col(j) - x_center_(j)) / x_scale_(j);
                    } else {
                        x_.col(j) = arma::zeros(x_.n_rows);
                        // make scale(j) nonzero for rescaling
                        x_scale_(j) = - 1.0;
                    }
                }
            }
            return this;
        }

        inline Abclass* set_k(const unsigned int k)
        {
            k_ = k;
            km1_ = k - 1;
            set_vertex_matrix(k);
            return this;
        }

        inline Abclass* set_intercept(const bool intercept)
        {
            control_.intercept_ = intercept;
            return this;
        }

        inline Abclass* set_standardize(const bool standardize)
        {
            control_.standardize_ = standardize;
            return this;
        }

        inline Abclass* set_weight(const arma::vec& weight)
        {
            if (weight.n_elem != n_obs_) {
                control_.obs_weight_ = arma::ones(n_obs_);
            } else {
                control_.obs_weight_ = weight / arma::sum(weight) * dn_obs_;
            }
            return this;
        }

        // setter for group weights
        inline void set_group_weight(
            const arma::vec& group_weight = arma::vec()
            )
        {
            if (group_weight.n_elem > 0) {
                control_.group_weight_ = gen_group_weight(group_weight);
            } else {
                control_.group_weight_ = gen_group_weight(
                    control_.group_weight_);
            }
        }

        // linear predictor
        inline arma::mat linear_score(const arma::mat& beta,
                                      const T_x& x) const
        {
            arma::mat pred_mat;
            if (control_.intercept_) {
                pred_mat = x * beta.tail_rows(x.n_cols);
                pred_mat.each_row() += beta.row(0);
            } else {
                pred_mat = x * beta;
            }
            return pred_mat;
        }
        // class conditional probability
        inline arma::mat predict_prob(const arma::mat& pred_f) const
        {
            // pred_f: n x (k - 1) matrix
            // vertex_: k x (k - 1) matrix
            arma::mat out { pred_f * vertex_.t() }; // n x k
            out.each_col([&](arma::vec& a) {
                a = 1.0 / loss_derivative(a);
            });
            arma::vec row_sums { arma::sum(out, 1) };
            out.each_col() /= row_sums;
            return out;
        }
        // predict categories for predicted classification functions
        inline arma::uvec predict_y(const arma::mat& pred_f) const
        {
            // pred_f: n x (k - 1) matrix
            // vertex_: k x (k - 1) matrix
            arma::mat out { pred_f * vertex_.t() }; // n x k
            return arma::index_max(out, 1);
        }

        // accuracy for tuning by cross-validation
        inline double accuracy(const arma::mat& pred_f,
                               const arma::uvec& y) const
        {
            arma::uvec max_idx { predict_y(pred_f) };
            // note that y can be of length different than dn_obs_
            return arma::sum(max_idx == y) /
                static_cast<double>(y.n_elem);
        }
        // class conditional probability
        inline arma::mat predict_prob(const arma::mat& beta,
                                      const T_x& x) const
        {
            return predict_prob(linear_score(beta, x));
        }
        // prediction based on the inner products
        inline arma::uvec predict_y(const arma::mat& beta,
                                    const T_x& x) const
        {
            return predict_y(linear_score(beta, x));
        }
        // accuracy for tuning
        inline double accuracy(const arma::mat& beta,
                               const T_x& x,
                               const arma::uvec& y) const
        {
            return accuracy(linear_score(beta, x), y);
        }

    };

}



#endif /* ABCLASS_ABCLASS_H */
