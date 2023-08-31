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
                query_vec_.at(i).max_dcg();
                pairs_end_(i) = query_vec_.at(i).n_pairs_;
            }
            pairs_end_ = arma::cumsum(pairs_end_);
            pairs_start_ = arma::shift(pairs_end_, 1);
            n_all_pairs_ = pairs_start_(0);
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
        inline Abrank* set_lambda_weight(const arma::vec& preds,
                                         const bool balance_query = true)
        {
            if (preds.n_elem != n_all_pairs_) {
                throw std::range_error("The preds must match the queries.");
            }
            arma::vec out_w { arma::ones(n_all_pairs_) };
            for (size_t i {0}; i < query_vec_.size(); ++i) {
                arma::vec pred_i {
                    preds.subvec(pairs_start_(i), pairs_end_(i))
                };
                arma::vec w_vec { query_vec_.at(i).delta_dcg(pred_i) };
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

        // train the classifier
        inline Abrank* fit()
        {
            abc_.fit();
            return this;
        }

        // the prediction method
        inline arma::mat predict(const arma::mat& beta,
                                 const T_x& x,
                                 const arma::mat& offset) const
        {
            return abc_.linear_score(beta, x, offset);
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

    // cross-validation
    template <typename T_loss, typename T_x>
    inline arma::cube cv_abrank_recall(const std::vector<T_x>& xs,
                                       const std::vector<arma::vec>& ys,
                                       const Control& control,
                                       const arma::vec& top_props)
    {
        size_t ntune { control.lambda_.n_elem };
        if (ntune == 0) {
            ntune = control.nlambda_;
        }
        arma::cube cv_recall = arma::zeros(top_props.n_elem, ntune, xs.size());
        for (size_t i {0}; i < xs.size(); ++i) {
            T_x test_x { xs.at(i) };
            arma::vec test_y { ys.at(i) };
            arma::vec test_offset { arma::zeros(test_x.n_rows) };
            Query<T_x> test_query { test_y };
            std::vector<T_x> train_xs { xs };
            std::vector<arma::vec> train_ys { ys };
            train_xs.erase(train_xs.begin() + i);
            train_ys.erase(train_ys.begin() + i);
            Abrank<T_loss, T_x> cv_obj { train_xs, train_ys };
            cv_obj.fit();
            for (size_t j {0}; j < ntune; ++j) {
                arma::vec cv_pred {
                    mat2vec(cv_obj.predict(cv_obj.abc_.coef_.slice(j),
                                           test_x, test_offset))
                };
                cv_recall.slice(i).col(j) =
                    test_query.recall(cv_pred, top_props, false);
            }
        }
        return cv_recall;
    }

}  // abclass


#endif /* ABCLASS_ABRANK_H */
