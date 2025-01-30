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

#ifndef ABCLASS_LIKEBOOST_H
#define ABCLASS_LIKEBOOST_H

#include <RcppArmadillo.h>
#include "Simplex.h"

namespace abclass
{
    // negative log-likelihood of mlogit
    class LikeBoost
    {
    public:
        LikeBoost() {}

        // default to exponential loss
        inline virtual double neg_inv_d1loss(const double u) const
        {
            // - 1 / l'(u)
            return std::exp(u);
        }
        inline virtual double neg_d2_over_d1(const double u) const
        {
            // - l''(u) / l'(u)
            return 1.0;
        }
        inline virtual double d2_over_d1s(const double u) const
        {
            // l''(u) / (l'(u) ^ 2)
            return std::exp(u);
        }

        // MM lowerbound factor
        inline virtual double mm_lowerbound(const double dk) const
        {
            return 1.0 / dk;
        }

        // loss function with observational weights
        inline double loss(const arma::mat& pred_f0,
                           const arma::vec& obs_weight,
                           const arma::uvec& y) const
        {
            double res { 0.0 };
            for (size_t i {0}; i < y.n_elem; ++i) {
                arma::rowvec fi { pred_f0.row(i) };
                double numer { neg_inv_d1loss(fi[y[i]])  };
                double denom { numer };
                for (size_t j {0}; j < fi.n_elem; ++j) {
                    if (j == y[i]) {
                        continue;
                    }
                    denom += neg_inv_d1loss(fi[j]);
                }
                res += obs_weight(i) * (std::log(denom) - std::log(numer));
            }
            return res;
        }

        // probability score for the decision function of the k-th class
        inline arma::vec prob_score_k(const arma::vec& pred_k) const
        {
            arma::vec out_k { pred_k };
            for (size_t i {0}; i < pred_k.n_elem; ++i) {
                out_k[i] = neg_inv_d1loss(pred_k[i]);
            }
            return out_k;
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
                const arma::rowvec vi {
                    data.iter_pred_f_.row(i) * data.vertex_
                };
                double denom { 0.0 };
                for (size_t j {0}; j < vi.n_cols; ++j) {
                    denom += neg_inv_d1loss(vi(j));
                }
                for (size_t j {0}; j < data.t_vertex_.n_rows; ++j) {
                    double numer_j { d2_over_d1s(vi(j)) };
                    out.row(i) += (numer_j / denom) * data.t_vertex_.row(j);
                }
                out.row(i) -= neg_d2_over_d1(vi(data.y_[i])) *
                    data.t_vertex_.row(data.y_[i]);
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
                const arma::rowvec vi {
                    data.iter_pred_f_.row(i) * data.vertex_
                };
                double denom { 0.0 };
                for (size_t j {0}; j < vi.n_cols; ++j) {
                    denom += neg_inv_d1loss(vi(j));
                }
                for (size_t j {0}; j < data.t_vertex_.n_rows; ++j) {
                    double numer_j { d2_over_d1s(vi(j)) };
                    out(i) += (numer_j / denom) * data.t_vertex_(j, k);
                }
                out(i) -= neg_d2_over_d1(vi(data.y_[i])) *
                    data.t_vertex_(data.y_[i], k);
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

    };                          // end of class

}  // abclass

#endif /* ABCLASS_LIKEBOOST_H */
