#ifndef ABCLASS_QUERY_H
#define ABCLASS_QUERY_H

#include <cmath>
#include <vector>
#include <RcppArmadillo.h>

namespace abclass {

    class Query
    {
    protected:
        arma::uvec desc_idx_;   // order(y)
        arma::mat x_;           // sorted
        arma::vec y_;           // sorted in a descending order
        bool has_pairs_;        // if we have constructed the pairwise data

        // rank function in a descending order
        inline arma::uvec desc_rank(const arma::vec& pred) const
        {
            return arma::sort_index(arma::sort_index(pred, "descend"));
        }

    public:
        unsigned int n_pairs_;  // number of pairs
        arma::mat pair_x_;      // x[i, ] - x[j, ]
        arma::uvec pair_i_;
        arma::uvec pair_j_;

        arma::vec max_dcg_;
        arma::vec delta_dcg_;

        // constructors
        Query() {};

        explicit Query(const arma::vec& y)
        {
            desc_idx_ = arma::sort_index(y, "descend");
            y_ = y.elem(desc_idx_);
        }

        Query(const arma::mat& x,
              const arma::vec& y,
              const bool pairs = true)
        {
            desc_idx_ = arma::sort_index(y, "descend");
            y_ = y.elem(desc_idx_);
            x_ = x.rows(desc_idx_);
            if (pairs) {
                construct_pairs();
            }
        }

        void construct_pairs()
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
            pair_x_ = arma::zeros(ii, x_.n_cols);
            for (size_t i {0}; i < ii; ++i) {
                pair_x_.row(i) = x_.row(ivec[i]) - x_.row(jvec[i]);
            }
            has_pairs_ = true;
        }

        // methods
        inline double max_dcg(const unsigned int top_k = 1)
        {
            unsigned int k { std::max(1U, std::min(top_k, y_.n_elem)) };
            if (max_dcg_.n_elem == y_.n_elem) {
                return max_dcg_(k - 1);
            }
            max_dcg_ = arma::zeros(y_.n_elem);
            double out { 0.0 };
            for (size_t i {0}; i < k; ++i) {
                out += (std::pow(2, y_(i)) - 1) / std::log2(i + 2);
                max_dcg_(i) = out;
            }
            return out;
        }
        inline double dcg(const arma::vec& pred,
                          const unsigned int top_k = 1,
                          const bool sorted = true) const
        {
            arma::uvec pred_idx;

            if (! sorted) {
                pred_idx = arma::sort_index(pred.elem(desc_idx_), "descend");
            } else {
                pred_idx = arma::sort_index(pred, "descend");
            }
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
                           const bool sorted = true)
        {
            double max_dcg_k { max_dcg(top_k) };
            double dcg_k { dcg(pred, top_k, sorted) };
            return dcg_k / max_dcg_k;
        }

        // absolute value of dcg if swapping the pairs
        inline void compute_delta_dcg(const arma::vec& pred,
                                      const bool sorted = true)
        {
            if (! has_pairs_) {
                construct_pairs();
            }
            arma::uvec pred_drank;
            if (! sorted) {
                pred_drank = desc_rank(pred.elem(desc_idx_));
            } else {
                pred_drank = desc_rank(pred);
            }
            delta_dcg_ = arma::ones(n_pairs_);
            for (size_t i {0}; i < n_pairs_; ++i) {
                double g_i_p1 { std::pow(2, y_(pair_i_[i])) };
                double g_j_p1 { std::pow(2, y_(pair_j_[i])) };
                double d_i { std::log2(2.0 + pred_drank(pair_i_[i])) };
                double d_j { std::log2(2.0 + pred_drank(pair_j_[i])) };
                delta_dcg_(i) = std::abs((g_i_p1 - g_j_p1) *
                                         (1.0 / d_i - 1.0 / d_j));
            }
        }


    };

}  // abclass


#endif /* ABCLASS_QUERY_H */
