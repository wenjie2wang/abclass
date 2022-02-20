#ifndef ABCLASS_ABCLASS_H
#define ABCLASS_ABCLASS_H

#include <RcppArmadillo.h>
#include "Simplex.h"

namespace abclass
{
    // base class for the angle-based large margin classifiers
    class Abclass
    {
    protected:
        unsigned int n_obs_;    // number of observations
        unsigned int k_;        // number of categories
        unsigned int p0_;       // number of predictors without intercept
        arma::mat x_;           // (standardized) x_: n by p (with intercept)
        arma::uvec y_;          // y vector ranging in {0, ..., k - 1}
        arma::vec obs_weight_;  // optional observation weights: of length n
        arma::mat vertex_;      // unique vertex: k by (k - 1)
        bool intercept_;        // if to contrains intercepts
        bool standardize_;      // is x_ standardized (column-wise)
        arma::rowvec x_center_; // the column center of x_
        arma::rowvec x_scale_;  // the column scale of x_

        // cache variables
        double dn_obs_;              // double version of n_obs_
        unsigned int km1_;           // k - 1
        unsigned int p1_;            // number of predictors (with intercept)
        unsigned int int_intercept_; // integer version of intercept_

        // prepare the vertex matrix
        inline void set_vertex_matrix(const unsigned int k)
        {
            Simplex sim { k };
            vertex_ = sim.get_vertex();
        }

        inline arma::vec get_vertex_y(const unsigned int j) const
        {
            // j in {0, 1, ..., k - 2}
            arma::vec vj { vertex_.col(j) };
            return vj.elem(y_);
        }

        // transfer coef for standardized data to coef for non-standardized data
        inline arma::mat rescale_coef(const arma::mat& beta) const
        {
            arma::mat out { beta };
            if (standardize_) {
                if (intercept_) {
                    // for each columns
                    for (size_t k { 0 }; k < km1_; ++k) {
                        arma::vec coef_k { beta.col(k) };
                        out(0, k) = beta(0, k) -
                            arma::as_scalar((x_center_ / x_scale_) *
                                            coef_k.tail_rows(p0_));
                        for (size_t l { 1 }; l < p1_; ++l) {
                            out(l, k) = coef_k(l) / x_scale_(l - 1);
                        }
                    }
                } else {
                    for (size_t k { 0 }; k < km1_; ++k) {
                        for (size_t l { 0 }; l < p0_; ++l) {
                            out(l, k) /= x_scale_(l);
                        }
                    }
                }
            }
            return out;
        }

        // transfer coef for non-standardized data to coef for standardized data
        inline arma::vec rev_rescale_coef(const arma::vec& beta) const
        {
            arma::vec beta0 { beta };
            double tmp {0};
            for (size_t j {1}; j < beta.n_elem; ++j) {
                beta0(j) *= x_scale_(j - 1);
                tmp += beta(j) * x_center_(j - 1);
            }
            beta0(0) += tmp;
            return beta0;
        }

        // the negative first derivative of the loss function
        inline virtual arma::vec
        neg_loss_derivative(const arma::vec& inner) const = 0;

        // the first derivative of the loss function
        inline arma::vec loss_derivative(const arma::vec& inner) const
        {
            return - neg_loss_derivative(inner);
        }


    public:

        // default constructor
        Abclass() {}

        // for using prediction functions
        explicit Abclass(const unsigned k)
        {
            set_vertex_matrix(k);
            k_ = k;
        }

        // main constructor
        Abclass(const arma::mat& x,
                const arma::uvec& y,
                const bool intercept = true,
                const bool standardize = true,
                const arma::vec& weight = arma::vec()) :
            x_ (x),
            y_ (y),
            intercept_ (intercept),
            standardize_ (standardize)
        {
            int_intercept_ = static_cast<unsigned int>(intercept_);
            km1_ = arma::max(y_); // assume y in {0, ..., k-1}
            k_ = km1_ + 1;
            n_obs_ = x_.n_rows;
            dn_obs_ = static_cast<double>(n_obs_);
            p0_ = x_.n_cols;
            p1_ = p0_ + int_intercept_;
            if (weight.n_elem != n_obs_) {
                obs_weight_ = arma::ones(n_obs_);
            } else {
                obs_weight_ = weight / arma::sum(weight) * dn_obs_;
            }
            if (standardize_) {
                if (intercept_) {
                    x_center_ = arma::mean(x_);
                } else {
                    x_center_ = arma::zeros<arma::rowvec>(x_.n_cols);
                }
                x_scale_ = arma::stddev(x_, 1);
                for (size_t j {0}; j < p0_; ++j) {
                    if (x_scale_(j) > 0) {
                        x_.col(j) = (x_.col(j) - x_center_(j)) / x_scale_(j);
                    } else {
                        x_.col(j) = arma::zeros(x_.n_rows);
                        // make scale(j) nonzero for rescaling
                        x_scale_(j) = - 1.0;
                    }
                }
            }
            if (intercept_) {
                x_ = arma::join_horiz(arma::ones(n_obs_), x_);
            }
            // set vertex matrix
            set_vertex_matrix(k_);
        }

        // class conditional probability
        inline arma::mat predict_prob(const arma::mat& pred_f) const
        {
            // pred_f: n x (k - 1) matrix
            // vertex_: k x (k - 1) matrix
            arma::mat out { pred_f * vertex_.t() }; // n x k
            out.each_col([&](arma::vec& a) {
                a = 1.0 / loss_derivative(a);
            });
            arma::vec row_sums { arma::sum(out, 1) };
            out.each_col() /= row_sums;
            return out;
        }

        // predict categories for given probability matrix
        inline arma::uvec predict_y(const arma::mat& prob_mat) const
        {
            return arma::index_max(prob_mat, 1);
        }

        // accuracy for tuning by cross-validation
        inline double accuracy(const arma::mat& pred_f,
                               const arma::uvec& y) const
        {
            arma::mat prob_mat { predict_prob(pred_f) };
            arma::uvec max_idx { predict_y(prob_mat) };
            return arma::mean(max_idx == y);
        }

        // weights may be adjusted internally
        inline arma::vec get_weight() const
        {
            return obs_weight_;
        }


    };

}



#endif /* ABCLASS_ABCLASS_H */
