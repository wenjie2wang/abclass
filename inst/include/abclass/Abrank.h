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

#ifndef ABCLASS_ABRANK_H
#define ABCLASS_ABRANK_H

#include <vector>
#include <RcppArmadillo.h>

#include "AbclassNet.h"
#include "Control.h"
#include "Query.h"
#include "Logistic.h"
#include "Boost.h"
#include "HingeBoost.h"
#include "Lum.h"

namespace abclass
{
    template <typename T_loss, typename T_x>
    class Abrank
    {
    protected:
        std::vector<Query<T_x> > query_vec_;
        // cache
        size_t n_all_pairs_;
        // index variables
        arma::uvec pairs_start_;
        arma::uvec pairs_end_;

    public:
        AbclassNet<T_loss, T_x> abc_;

        // constructors
        Abrank() {};

        Abrank(const std::vector<T_x>& xs,
               const std::vector<arma::vec>& ys,
               Control control = Control())
        {
            if (xs.size() != ys.size()) {
                throw std::range_error("xs.size() must match ys.size()");
            }
            pairs_start_ = arma::zeros<arma::uvec>(xs.size());
            pairs_end_ = pairs_start_;
            for (size_t i {0}; i < xs.size(); ++i) {
                query_vec_.push_back(Query<T_x>(xs.at(i), ys.at(i)));
                query_vec_.at(i).compute_max_dcg();
                pairs_end_(i) = query_vec_.at(i).n_pairs_;
            }
            if (xs.size() > 1) {
                pairs_end_ = arma::cumsum(pairs_end_);
                pairs_start_ = arma::shift(pairs_end_, 1);
            }
            n_all_pairs_ = pairs_end_(xs.size() - 1);
            pairs_start_(0) = 0;
            pairs_end_ -= 1;
            T_x abc_x { query_vec_.at(0).pair_x_ };
            for (size_t i {1}; i < query_vec_.size(); ++i) {
                abc_x = arma::join_cols(abc_x, query_vec_.at(i).pair_x_);
            }
            arma::uvec abc_y { arma::zeros<arma::uvec>(n_all_pairs_) };
            // not need intercept
            control.set_intercept(false);
            abc_ = AbclassNet<T_loss, T_x>(abc_x, abc_y, control);
        }

        // set adaptive lambda-loss weights
        inline Abrank* set_delta_weight(const bool balance_query = true)
        {
            arma::vec out_w { arma::ones(n_all_pairs_) };
            for (size_t i {0}; i < query_vec_.size(); ++i) {
                arma::vec pred_i {
                    abc_.linear_score(abc_.coef_.slice(0),
                                      query_vec_.at(i).pair_x_)
                };
                arma::vec w_vec { query_vec_.at(i).delta_ndcg(pred_i) };
                if (balance_query) {
                    w_vec /= arma::accu(w_vec);
                }
                out_w.subvec(pairs_start_(i), pairs_end_(i)) = w_vec;
            }
            abc_.set_weight(out_w);
            return this;
        }

        // set query weights only
        inline Abrank* set_query_weight()
        {
            arma::vec out_w { arma::ones(n_all_pairs_) };
            for (size_t i {0}; i < query_vec_.size(); ++i) {
                out_w.subvec(pairs_start_(i), pairs_end_(i)) /=
                    query_vec_.at(i).n_pairs_;
            }
            abc_.set_weight(out_w);
            return this;
        }

        // the prediction method
        inline arma::mat predict(const arma::mat& beta,
                                 const T_x& x,
                                 const arma::mat& offset = arma::vec()) const
        {
            return abc_.linear_score(beta, x, offset);
        }

        // train the classifier
        inline Abrank* fit()
        {
            if (abc_.control_.query_weight_) {
                set_query_weight();
            }
            abc_.fit();
            if (abc_.control_.delta_weight_) {
                // assume there is only one lambda
                abc_.control_.lambda_ = abc_.control_.lambda_[0];
                set_delta_weight(abc_.control_.query_weight_);
                arma::vec w0 { abc_.control_.obs_weight_ };
                for (size_t i {0}; i < abc_.control_.delta_max_iter_; ++i) {
                    abc_.fit();
                    set_delta_weight(abc_.control_.query_weight_);
                    arma::vec w1 { abc_.control_.obs_weight_ };
                    double tol { rel_diff(w0, w1) };
                    if (tol < abc_.control_.epsilon_) {
                        break;
                    }
                    w0 = w1;
                }
            }
            return this;
        }

        // cross-validation
        inline arma::cube cv_abrank_recall(
            const arma::vec& top_props = {0.05, 0.10, 0.15, 0.25, 0.50}
            )
        {
            size_t ntune { abc_.control_.lambda_.n_elem };
            if (ntune == 0) {
                ntune = abc_.control_.nlambda_;
            }
            arma::cube cv_recall {
                arma::zeros(top_props.n_elem, ntune, query_vec_.size())
            };
            for (size_t i {0}; i < query_vec_.size(); ++i) {
                T_x test_x { query_vec_.at(i).x_ };
                Query<T_x> test_query { query_vec_.at(i).y_ };
                std::vector<T_x> train_xs;
                std::vector<arma::vec> train_ys;
                for (size_t k {0}; k < query_vec_.size(); ++k) {
                    if (k == i) {
                        continue;
                    }
                    train_xs.push_back(query_vec_.at(k).x_);
                    train_ys.push_back(query_vec_.at(k).y_);
                }
                Abrank<T_loss, T_x> cv_obj {
                    train_xs, train_ys, abc_.control_
                };
                cv_obj.fit();
                for (size_t j {0}; j < ntune; ++j) {
                    arma::vec cv_pred {
                        mat2vec(cv_obj.predict(cv_obj.abc_.coef_.slice(j),
                                               test_x))
                    };
                    cv_recall.slice(i).col(j) =
                        test_query.recall(cv_pred, top_props, false);
                }
            }
            return cv_recall;
        }

    };

    // aliases
    template<typename T_x>
    using LogisticRank = abclass::Abrank<abclass::Logistic, T_x>;

    template<typename T_x>
    using BoostRank = abclass::Abrank<abclass::Boost, T_x>;

    template<typename T_x>
    using HingeBoostRank = abclass::Abrank<abclass::HingeBoost, T_x>;

    template<typename T_x>
    using LumRank = abclass::Abrank<abclass::Lum, T_x>;

}  // abclass


#endif /* ABCLASS_ABRANK_H */
