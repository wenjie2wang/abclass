#ifndef CROSS_VALIDATION_H
#define CROSS_VALIDATION_H

#include <vector>
#include <RcppArmadillo.h>
#include "utils.h"

namespace Malc {

    class CrossValidation {
    private:
        unsigned long n_obs_;
        unsigned long n_folds_ = 10;

        // generate cross-validation indices
        // for given number of folds and number of observations
        inline std::vector<arma::uvec> get_cv_test_index(
            const unsigned long n_obs,
            const unsigned long n_folds = 10
            )
        {
            // number of observations must be at least two
            if (n_obs < 2) {
                throw std::range_error(
                    "Cross-validation needs at least two observations."
                    );
            }
            // number of folds is at most number of observations
            if (n_folds > n_obs) {
                throw std::range_error(
                    "Number of folds should be <= number of observations."
                    );
            }
            // define output
            std::vector<arma::uvec> out;
            // observation indices random permuted
            arma::uvec obs_idx { arma::randperm(n_obs) };
            // remaining number of observations
            size_t re_n_obs { n_obs };
            // determine the size of folds and indices one by one
            for (size_t i {0}; i < n_folds; ++i) {
                size_t fold_i { re_n_obs / (n_folds - i) };
                size_t j { n_obs - re_n_obs };
                arma::uvec idx_i { obs_idx.subvec(j, j + fold_i - 1) };
                out.push_back(idx_i);
                re_n_obs -= fold_i;
            }
            return out;
        }

    public:
        std::vector<arma::uvec> train_index;
        std::vector<arma::uvec> test_index;

        // default constructor
        CrossValidation();

        // major constructor
        CrossValidation(const unsigned long n_obs,
                        const unsigned long n_folds) :
            n_obs_ { n_obs },
            n_folds_ { n_folds }
        {
            test_index = get_cv_test_index(n_obs_, n_folds_);
            arma::uvec all_index {
                arma::regspace<arma::uvec>(0, n_obs - 1)
            };
            for (size_t i {0}; i < n_folds_; ++i) {
                train_index.push_back(
                    vec_diff(all_index, test_index.at(i))
                    );
            }
        }

        // explicit constructor
        explicit CrossValidation(const unsigned long n_obs) :
            n_obs_ { n_obs }
        {
            CrossValidation(n_obs_, 10);
        }

        // strata takes values from {0, ..., k}
        CrossValidation(const unsigned long n_obs,
                        const unsigned long n_folds,
                        const arma::uvec& strata) :
            n_obs_ { n_obs },
            n_folds_ { n_folds }
        {
            const unsigned int n_strata { arma::max(strata) + 1 };
            // for the first strata
            arma::uvec k_idx { arma::find(strata == 0) };
            unsigned int n_k { k_idx.n_elem };
            CrossValidation cv_obj { n_k, n_folds };
            for (size_t ii { 0 }; ii < n_folds; ++ii) {
                train_index.push_back(k_idx.elem(cv_obj.train_index.at(ii)));
                test_index.push_back(k_idx.elem(cv_obj.test_index.at(ii)));
            }
            // for the remaining strata
            for (size_t j { 1 }; j < n_strata; ++j) {
                k_idx = arma::find(strata == j);
                n_k = k_idx.n_elem;
                cv_obj = CrossValidation(n_k, n_folds);
                for (size_t ii { 0 }; ii < n_folds; ++ii) {
                    train_index.at(ii) = arma::join_cols(
                        train_index.at(ii),
                        k_idx.elem(cv_obj.train_index.at(ii))
                        );
                    test_index.at(ii) = arma::join_cols(
                        test_index.at(ii),
                        k_idx.elem(cv_obj.test_index.at(ii))
                        );
                }
            }
        }

        // helper function
        unsigned long get_n_folds() const
        {
            return n_folds_;
        }
        unsigned long get_n_obs() const
        {
            return n_obs_;
        }

    };
}  // Malc


#endif /* CROSS_VALIDATION_H */
