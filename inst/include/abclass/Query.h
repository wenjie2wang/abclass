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

#ifndef ABCLASS_QUERY_H
#define ABCLASS_QUERY_H

#include <cmath>
#include <vector>
#include <RcppArmadillo.h>

namespace abclass {

    template <typename T_x = arma::mat>
    class Query
    {
    protected:
        arma::uvec desc_idx_;   // order(y, descending = TRUE)
        arma::uvec asc_idx_;    // order(y, descending = FALSE)

        // align predictions with y_
        inline arma::uvec get_pred_idx(const arma::vec& pred,
                                       const bool sorted = true) const
        {
            arma::uvec pred_idx;
            if (sorted) {
                pred_idx = arma::stable_sort_index(pred.elem(asc_idx_),
                                                   "descend");
            } else {
                arma::vec tmp_pred { pred.elem(desc_idx_) };
                pred_idx = arma::stable_sort_index(tmp_pred.elem(asc_idx_),
                                                   "descend");
            }
            return asc_idx_.elem(pred_idx);
        }

    public:
        arma::vec y_;           // sorted in a descending order
        T_x x_;                 // sorted corresponding to y

        bool has_pairs_;        // if we have constructed the pairwise data
        unsigned int n_pairs_;  // number of pairs
        T_x pair_x_;            // x[i, ] - x[j, ]
        arma::uvec pair_i_;
        arma::uvec pair_j_;

        arma::vec max_dcg_;

        // constructors
        Query() {};

        explicit Query(const arma::vec& y,
                       const bool pairs = false)
        {
            desc_idx_ = arma::sort_index(y, "descend");
            asc_idx_ = arma::sort_index(y, "ascend");
            y_ = y.elem(desc_idx_);
            if (pairs) {
                construct_pairs(false);
            }
        }

        Query(const T_x& x,
              const arma::vec& y,
              const bool pairs = true)
        {
            desc_idx_ = arma::sort_index(y, "descend");
            asc_idx_ = arma::sort_index(y, "ascend");
            y_ = y.elem(desc_idx_);
            x_ = x.rows(desc_idx_);
            if (pairs) {
                construct_pairs(true);
            }
        }

        // methods
        inline Query* construct_pairs(const bool with_x = true)
        {
            size_t ii {0};
            std::vector<unsigned int> ivec, jvec;
            size_t npairs { y_.n_elem * (y_.n_elem + 1) / 2 };
            ivec.reserve(npairs);
            jvec.reserve(npairs);
            for (size_t i {0}; i < y_.n_elem; ++i) {
                for (size_t j {0}; j < y_.n_elem; ++j) {
                    if (i == j) {
                        continue;
                    }
                    double tmp { y_[i] - y_[j] };
                    if (tmp > 0.0) {
                        ivec.push_back(i);
                        jvec.push_back(j);
                        ++ii;
                    }
                }
            }
            n_pairs_ = ii;
            pair_i_ = arma::uvec(ivec);
            pair_j_ = arma::uvec(jvec);
            if (with_x) {
                pair_x_ = arma::zeros(ii, x_.n_cols);
                for (size_t i {0}; i < ii; ++i) {
                    pair_x_.row(i) = x_.row(ivec[i]) - x_.row(jvec[i]);
                }
            }
            has_pairs_ = true;
            return this;
        }

        inline Query* compute_max_dcg()
        {
            max_dcg_ = arma::zeros(y_.n_elem);
            double tmp { 0.0 };
            for (size_t i {0}; i < y_.n_elem; ++i) {
                tmp += (std::pow(2, y_(i)) - 1) / std::log2(i + 2);
                max_dcg_(i) = tmp;
            }
            return this;
        }

        // rank function in a descending order
        inline arma::uvec desc_rank(const arma::vec& pred) const
        {
            return arma::sort_index(arma::sort_index(pred, "descend"));
        }

        inline double max_dcg(const unsigned int top_k = 1) const
        {
            unsigned int k { std::max(1U, std::min(top_k, y_.n_elem)) };
            double out { 0.0 };
            for (size_t i {0}; i < k; ++i) {
                out += (std::pow(2, y_(i)) - 1) / std::log2(i + 2);
            }
            return out;
        }

        inline double dcg(const arma::vec& pred,
                          const unsigned int top_k = 1,
                          const bool sorted = true) const
        {
            arma::uvec pred_idx { get_pred_idx(pred, sorted) };
            arma::vec score { y_.elem(pred_idx) };
            unsigned int k { std::min(top_k, y_.n_elem) };
            double out { 0.0 };
            for (size_t i {0}; i < k; ++i) {
                out += (std::pow(2, score(i)) - 1) / std::log2(i + 2);
            }
            return out;
        }

        inline double ndcg(const arma::vec& pred,
                           const unsigned int top_k = 1,
                           const bool sorted = true) const
        {
            double max_dcg_k { max_dcg(top_k) };
            double dcg_k { dcg(pred, top_k, sorted) };
            return dcg_k / max_dcg_k;
        }

        // absolute value of ndcg if swapping the pairs
        inline arma::vec delta_ndcg(const arma::vec& pred,
                                    const bool sorted = true)
        {
            if (! has_pairs_) {
                construct_pairs(false);
                compute_max_dcg();
            }
            arma::uvec pred_drank;
            if (! sorted) {
                pred_drank = desc_rank(pred.elem(desc_idx_));
            } else {
                pred_drank = desc_rank(pred);
            }
            arma::vec out { arma::ones(n_pairs_) };
            for (size_t i {0}; i < n_pairs_; ++i) {
                double g_i_p1 { std::pow(2, y_(pair_i_[i])) };
                double g_j_p1 { std::pow(2, y_(pair_j_[i])) };
                double d_i { std::log2(2.0 + pred_drank(pair_i_[i])) };
                double d_j { std::log2(2.0 + pred_drank(pair_j_[i])) };
                out(i) = std::abs((g_i_p1 - g_j_p1) * (1.0 / d_i - 1.0 / d_j));
            }
            double max_dcg_k { max_dcg_(y_.n_elem - 1) };
            return out / max_dcg_k;
        }
        inline arma::vec delta_ndcg()
        {
            return delta_ndcg(y_, true);
        }

        // recall
        inline arma::vec recall(const arma::vec& pred,
                                const arma::vec& top_props,
                                const bool sorted = true) const
        {
            // we sort y in an ascending order to break ties
            // corresponds to the recall of the worst case
            arma::uvec pred_idx { get_pred_idx(pred) };
            arma::vec out { arma::zeros(top_props.n_elem) };
            for (size_t i {0}; i < top_props.n_elem; ++i) {
                unsigned int k {
                    static_cast<unsigned int>(
                        std::floor(top_props[i] * y_.n_elem))
                };
                k = std::max(1U, k);
                out[i] = arma::accu(pred_idx.head(k) < k) /
                    static_cast<double>(k);
            }
            return out;
        }
        // delta recall compared to expected recall by random
        inline arma::vec delta_recall(const arma::vec& pred,
                                      const arma::vec& top_props,
                                      const bool sorted = true) const
        {
            return recall(pred, top_props, sorted) - top_props;
        }
        // sum of delta recall up to top k%
        inline double delta_recall_sum(const arma::vec& pred,
                                       const double until_top) const
        {
            arma::vec top_props { arma::regspace(0.01, 0.01, until_top) };
            arma::vec tmp_delta { delta_recall(pred, top_props, true) };
            return arma::accu(tmp_delta - top_props);
        }

        // generate pairwise offsets based on predictions for Abrank
        inline arma::vec abrank_offset(const arma::vec& pred,
                                       const bool sorted = true) const
        {
            if (pred.n_elem != y_.n_elem) {
                throw std::range_error(
                    "The length of predictions must match the data.");
            }
            arma::vec out { arma::ones(n_pairs_) };
            for (size_t i {0}; i < n_pairs_; ++i) {
                out(i) = pred(pair_i_[i]) - pred(pair_j_[i]);
            }
            return out;
        }

        inline arma::uvec get_rev_idx() const
        {
            return arma::sort_index(desc_idx_);
        }

    };

}  // abclass


#endif /* ABCLASS_QUERY_H */
