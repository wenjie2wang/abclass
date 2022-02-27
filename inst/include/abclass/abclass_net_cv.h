#ifndef ABCLASS_ABCLASS_NET_CV_H
#define ABCLASS_ABCLASS_NET_CV_H

#include <utility>
#include <RcppArmadillo.h>
#include "AbclassNet.h"
#include "CrossValidation.h"
#include "utils.h"

namespace abclass {

    // cross-validation method for AbclassNet objects
    template <typename T>
    inline void abclass_net_cv(T& obj,
                               const unsigned int nfolds = 5,
                               const arma::uvec strata = arma::uvec())
    {
        CrossValidation cv_obj { obj.n_obs_, nfolds, strata };
        obj.cv_accuracy_ = arma::zeros(obj.lambda_.n_elem, nfolds);
        // model fits
        for (size_t i { 0 }; i < nfolds; ++i) {
            arma::mat train_x { obj.x_.rows(cv_obj.train_index_.at(i)) };
            if (obj.intercept_) {
                train_x = train_x.tail_cols(obj.p0_);
            }
            arma::uvec train_y { obj.y_.rows(cv_obj.train_index_.at(i)) };
            arma::vec train_weight {
                obj.obs_weight_.rows(cv_obj.train_index_.at(i))
            };
            arma::mat test_x { obj.x_.rows(cv_obj.test_index_.at(i)) };
            arma::uvec test_y { obj.y_.rows(cv_obj.test_index_.at(i)) };
            // create a new object
            T new_obj { obj };
            new_obj.set_data(std::move(train_x),
                             std::move(train_y));
            new_obj.set_weight(std::move(train_weight));
            new_obj.fit(obj.lambda_, obj.alpha_, 0, 1,
                        obj.max_iter_, obj.rel_tol_,
                        obj.varying_active_set_, 0);
            for (size_t l { 0 }; l < obj.lambda_.n_elem; ++l) {
                obj.cv_accuracy_(l, i) = new_obj.accuracy(
                    new_obj.coef_.slice(l), test_x, test_y);
            }
        }
        obj.cv_accuracy_mean_ = mat2vec(arma::mean(obj.cv_accuracy_, 1));
        obj.cv_accuracy_sd_ = mat2vec(arma::stddev(obj.cv_accuracy_, 0, 1));
    }


}

#endif
