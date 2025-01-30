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

#ifndef ABCLASS_MLOGIT_H
#define ABCLASS_MLOGIT_H

#include <RcppArmadillo.h>
#include "Simplex.h"

namespace abclass
{
    // negative log-likelihood of multinomial logistic model
    class Mlogit
    {
    public:
        Mlogit() {}

        // loss function with observational weights
        inline double loss(const arma::mat& pred_f0,
                           const arma::vec& obs_weight,
                           const arma::uvec& y) const
        {
            double res { 0.0 };
            for (size_t i {0}; i < y.n_elem; ++i) {
                arma::rowvec fi { pred_f0.row(i) };
                fi -= fi(y(i));
                double tmp { 0.0 };
                for (size_t j {0}; j < fi.n_elem; ++j) {
                    if (j == y[i]) {
                        tmp += 1.0;
                        continue;
                    }
                    tmp += std::exp(fi[j]);
                }
                res += obs_weight(i) * std::log(tmp);
            }
            return res;
        }

        // probability score for the decision function of the k-th class
        inline arma::vec prob_score_k(const arma::vec& pred_k) const
        {
            return arma::exp(pred_k);
        }

        // methods for Abclass
        template<typename T_x>
        inline double loss(const Simplex2<T_x>& data,
                           const arma::vec& obs_weight) const
        {
            arma::mat pred_f0 { data.iter_pred_f_ * data.vertex_ };
            return loss(pred_f0, obs_weight, data.y_);
        }

        // gradient of loss wrt the (K-1) decision functions with weights
        template<typename T_x>
        inline arma::mat dloss_df(const Simplex2<T_x>& data,
                                  const arma::vec& obs_weight) const
        {
            arma::mat out(data.n_obs_, data.km1_);
            for (size_t i {0}; i < data.n_obs_; ++i) {
                arma::rowvec vi { data.iter_pred_f_.row(i) * data.vertex_ };
                vi = arma::exp(vi);
                vi /= arma::accu(vi);
                for (size_t j {0}; j < data.t_vertex_.n_rows; ++j) {
                    if (j == data.y_[i]) {
                        continue;
                    }
                    arma::rowvec w_diff {
                        data.t_vertex_.row(j) - data.t_vertex_.row(data.y_[i])
                    };
                    out.row(i) += vi(j) * w_diff;
                }
                out.row(i) *= obs_weight(i);
            }
            return out;
        }

        // gradient of loss wrt the k-th decision function
        template<typename T_x>
        inline arma::vec dloss_df(const Simplex2<T_x>& data,
                                  const arma::vec& obs_weight,
                                  const unsigned int k) const
        {
            arma::vec out(data.y_.n_elem);
            for (size_t i {0}; i < data.y_.n_elem; ++i) {
                arma::rowvec vi { data.iter_pred_f_.row(i) * data.vertex_ };
                vi = arma::exp(vi);
                vi /= arma::accu(vi);
                for (size_t j {0}; j < data.t_vertex_.n_rows; ++j) {
                    if (j == data.y_[i]) {
                        continue;
                    }
                    double w_diff {
                        data.t_vertex_(j, k) - data.t_vertex_(data.y_[i], k)
                    };
                    out(i) += vi(j) * w_diff;
                }
                out(i) *= obs_weight(i);
            }
            return out;
        }

        // for linear learning
        // gradient wrt beta_g.
        template <typename T_x>
        inline arma::mat dloss_dbeta(const Simplex2<T_x>& data,
                                     const arma::vec& obs_weight,
                                     const unsigned int g) const
        {
            arma::mat dmat { dloss_df(data, obs_weight) };
            for (size_t j {0}; j < dmat.n_cols; ++j) {
                dmat.col(j) %= data.x_.col(g);
            }
            return dmat;
        }

        // gradient wrt beta_gk
        template <typename T_x>
        inline arma::vec dloss_dbeta(const Simplex2<T_x>& data,
                                     const arma::vec& obs_weight,
                                     const unsigned int g,
                                     const unsigned int k) const
        {
            arma::vec dvec { dloss_df(data, obs_weight, k) };
            dvec %= data.x_.col(g);
            return dvec;
        }

        // MM lowerbound factor
        inline double mm_lowerbound(const double dk) const
        {
            return 1.0 / dk;
        }

    };                          // end of class

}  // abclass

#endif /* ABCLASS_MLOGIT_H */
