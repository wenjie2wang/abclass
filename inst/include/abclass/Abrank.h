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
        std::vector<arma::vec> offset_vec_;
        // index variables
        arma::uvec pairs_start_;
        arma::uvec pairs_end_;

    public:
        AbclassNet<T_loss, T_x> abc_;

        // constructors
        Abrank() {};

        Abrank(const T_x& x,
               const arma::vec& y,
               const arma::uvec& qid,
               Control control = Control())
        {
            // qid must take values in {0, 1, 2, ...}
            const size_t nquery { qid.max() + 1 };
            pairs_start_ = arma::zeros<arma::uvec>(nquery);
            pairs_end_ = pairs_start_;
            arma::vec abc_offset;
            for (size_t i {0}; i < nquery; ++i) {
                arma::uvec is_i { arma::find(qid == i) };
                arma::mat x_i { x.rows(is_i) };
                arma::vec y_i { y.elem(is_i) };
                query_vec_.push_back(Query<T_x>(x_i, y_i));
                query_vec_.at(i).compute_max_dcg();
                pairs_end_(i) = query_vec_.at(i).n_pairs_;
                if (control.has_offset_) {
                    offset_vec_.push_back(control.offset_.elem(is_i));
                    abc_offset = arma::join_cols(
                        abc_offset,
                        query_vec_.at(i).abrank_offset(offset_vec_.at(i)));
                } else {
                    offset_vec_.push_back(arma::vec());
                }
            }
            if (nquery > 1) {
                pairs_end_ = arma::cumsum(pairs_end_);
                pairs_start_ = arma::shift(pairs_end_, 1);
            }
            n_all_pairs_ = pairs_end_(nquery - 1);
            pairs_start_(0) = 0;
            pairs_end_ -= 1;
            T_x abc_x { query_vec_.at(0).pair_x_ };
            for (size_t i {1}; i < query_vec_.size(); ++i) {
                abc_x = arma::join_cols(abc_x, query_vec_.at(i).pair_x_);
            }
            arma::uvec abc_y(n_all_pairs_);
            // not need intercept
            control.set_intercept(false);
            // set offset
            control.set_offset(abc_offset);
            abc_ = AbclassNet<T_loss, T_x>(abc_x, abc_y, control);
        }

        // set adaptive lambda-loss weights
        inline Abrank* set_delta_weight(const bool balance_query = true)
        {
            arma::vec out_w { arma::ones(n_all_pairs_) };
            for (size_t i {0}; i < query_vec_.size(); ++i) {
                arma::vec w_vec;
                if (abc_.control_.delta_adaptive_) {
                    arma::vec pred_i {
                        abc_.linear_score(abc_.coef_.slice(0),
                                          query_vec_.at(i).pair_x_)
                    };
                    w_vec = query_vec_.at(i).delta_ndcg(pred_i);
                } else {
                    w_vec = query_vec_.at(i).delta_ndcg();
                }
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
        inline arma::vec predict(const arma::vec& beta,
                                 const T_x& x,
                                 const arma::vec& offset = arma::vec()) const
        {
            return abc_.linear_score(beta, x, offset);
        }

        // train the classifier
        inline Abrank* fit()
        {
            if (abc_.control_.delta_weight_) {
                abc_.fit();
                // TODO only one lambda is allowed now
                abc_.control_.lambda_ = abc_.control_.lambda_(0);
                set_delta_weight(abc_.control_.query_weight_);

                if (! abc_.control_.delta_adaptive_) {
                    abc_.control_.delta_max_iter_ = 0;
                }
                arma::vec w0 { abc_.control_.obs_weight_ };
                for (size_t i {0}; i < abc_.control_.delta_max_iter_; ++i) {
                    set_delta_weight(abc_.control_.query_weight_);
                    arma::vec w1 { abc_.control_.obs_weight_ };
                    double tol { rel_diff(w0, w1) };
                    if (tol < abc_.control_.epsilon_) {
                        break;
                    }
                    w0 = w1;
                    abc_.fit();
                }
                return this;
            }
            if (abc_.control_.query_weight_) {
                set_query_weight();
            }
            abc_.fit();
            return this;
        }

        // cross-validation
        inline arma::cube cv_recall(
            const arma::vec& top_props = {0.05, 0.10, 0.15, 0.25, 0.50}
            )
        {
            size_t ntune { abc_.control_.lambda_.n_elem };
            if (ntune == 0) {
                ntune = abc_.control_.nlambda_;
            }
            arma::cube out_recall {
                arma::zeros(top_props.n_elem, ntune, query_vec_.size())
            };
            for (size_t i {0}; i < query_vec_.size(); ++i) {
                T_x test_x { query_vec_.at(i).x_ };
                Query<T_x> test_query { query_vec_.at(i).y_ };
                arma::vec test_offset { offset_vec_.at(i) };
                T_x train_x;
                arma::vec train_y;
                arma::uvec train_qid;
                arma::vec train_offset;
                for (size_t k {0}, ki {0}; k < query_vec_.size(); ++k) {
                    if (k == i) {
                        continue;
                    }
                    train_x = arma::join_cols(train_x, query_vec_.at(k).x_);
                    train_y = arma::join_cols(train_y, query_vec_.at(k).y_);
                    train_offset = arma::join_cols(train_offset,
                                                   offset_vec_.at(k));
                    train_qid = arma::join_cols(
                        train_qid,
                        arma::uvec(query_vec_.at(k).y_.n_elem,
                                   arma::fill::value(ki))
                        );
                    ++ki;
                }
                Control train_ctrl { abc_.control_ };
                train_ctrl.set_offset(train_offset);
                Abrank<T_loss, T_x> cv_obj {
                    train_x, train_y, train_qid, train_ctrl
                };
                cv_obj.fit();
                for (size_t j {0}; j < ntune; ++j) {
                    arma::vec cv_pred {
                        mat2vec(cv_obj.predict(cv_obj.abc_.coef_.slice(j),
                                               test_x, test_offset))
                    };
                    out_recall.slice(i).col(j) =
                        test_query.recall(cv_pred, top_props, false);
                }
            }
            return out_recall;
        }
        inline arma::mat cv_delta_recall(const double until_top = 0.50)
        {
            size_t ntune { abc_.control_.lambda_.n_elem };
            if (ntune == 0) {
                ntune = abc_.control_.nlambda_;
            }
            arma::mat out_recall {
                arma::zeros(query_vec_.size(), ntune)
            };
            for (size_t i {0}; i < query_vec_.size(); ++i) {
                T_x test_x { query_vec_.at(i).x_ };
                Query<T_x> test_query { query_vec_.at(i).y_ };
                arma::vec test_offset { offset_vec_.at(i) };
                T_x train_x;
                arma::vec train_y;
                arma::uvec train_qid;
                arma::vec train_offset;
                for (size_t k {0}, ki {0}; k < query_vec_.size(); ++k) {
                    if (k == i) {
                        continue;
                    }
                    train_x = arma::join_cols(train_x, query_vec_.at(k).x_);
                    train_y = arma::join_cols(train_y, query_vec_.at(k).y_);
                    train_offset = arma::join_cols(train_offset,
                                                   offset_vec_.at(k));
                    train_qid = arma::join_cols(
                        train_qid,
                        arma::uvec(query_vec_.at(k).y_.n_elem,
                                   arma::fill::value(ki))
                        );
                    ++ki;
                }
                Control train_ctrl { abc_.control_ };
                train_ctrl.set_offset(train_offset);
                Abrank<T_loss, T_x> cv_obj {
                    train_x, train_y, train_qid, train_ctrl
                };
                cv_obj.fit();
                for (size_t j {0}; j < ntune; ++j) {
                    arma::vec cv_pred {
                        mat2vec(cv_obj.predict(cv_obj.abc_.coef_.slice(j),
                                               test_x, test_offset))
                    };
                    out_recall(i, j) =
                        test_query.delta_recall_sum(cv_pred, until_top);
                }
            }
            return out_recall;
        }

    };

    // aliases
    template<typename T_x>
    using LogisticRank = Abrank<Logistic, T_x>;

    template<typename T_x>
    using BoostRank = Abrank<Boost, T_x>;

    template<typename T_x>
    using HingeBoostRank = Abrank<HingeBoost, T_x>;

    template<typename T_x>
    using LumRank = Abrank<Lum, T_x>;

}  // abclass


#endif /* ABCLASS_ABRANK_H */
