#ifndef ABCLASS_LOGISTIC_NET_CV_H
#define ABCLASS_LOGISTIC_NET_CV_H

#include <utility>
#include <vector>
#include <RcppArmadillo.h>
#include "utils.h"
#include "cross-validation.h"
#include "logistic_net.h"


namespace Abclass {

    class LogisticNetCV: public LogisticNet
    {
    protected:
        unsigned int nfolds_ = 5;
        bool stratified_ = true;
        arma::uvec strata_;
        CrossValidation cv_;

    public:
        arma::mat cv_accuracy_;

        // default constructor
        LogisticNetCV() = default;

        LogisticNetCV(const LogisticNet& object) :
            LogisticNet(object) {}

        inline LogisticNetCV* set(const unsigned int nfolds = 5,
                                  const arma::uvec strata = arma::uvec())
        {
            nfolds_ = nfolds;
            strata_ = strata;
            cv_ = CrossValidation(n_obs_, nfolds_, strata_);
            return this;
        }

        inline void tune(const unsigned int max_iter = 200,
                         const double rel_tol = 1e-4)
        {
            if (path_) {
                cv_accuracy_ = arma::zeros(lambda_path_.n_elem, nfolds_);
            } else {
                cv_accuracy_ = arma::zeros(1, nfolds_);
            }
            // model fits
            for (size_t i { 0 }; i < nfolds_; ++i) {
                arma::mat train_x { x_.rows(cv_.train_index_.at(i)) };
                if (intercept_) {
                    train_x = train_x.tail_cols(p0_);
                }
                arma::uvec train_y { y_.rows(cv_.train_index_.at(i)) };
                arma::mat test_x { x_.rows(cv_.test_index_.at(i)) };
                arma::uvec test_y { y_.rows(cv_.test_index_.at(i)) };
                LogisticNet reg_obj {
                    std::move(train_x), std::move(train_y),
                    intercept_, standardize_
                };
                if (path_) {
                    reg_obj.path(lambda_path_, alpha_, 0, 1,
                                 max_iter, rel_tol, pmin_, false);
                    for (size_t l { 0 }; l < lambda_path_.n_elem; ++l) {
                        cv_accuracy_(l, i) = reg_obj.accuracy(
                            reg_obj.coef_path_.slice(l), test_x, test_y);
                    }
                } else {
                    reg_obj.fit(lambda_, alpha_, arma::vec(), max_iter,
                                rel_tol, pmin_, false);
                    cv_accuracy_(0, i) = reg_obj.accuracy(
                        reg_obj.coef_, test_x, test_y);
                }
            }
        }

    };



}

#endif
