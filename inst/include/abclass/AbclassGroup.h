#ifndef ABCLASS_ABCLASS_GROUP_H
#define ABCLASS_ABCLASS_GROUP_H

#include <RcppArmadillo.h>
#include "Abclass.h"
#include "Control.h"

namespace abclass
{
    // base class for the angle-based large margin classifiers
    // with group-wise regularization

    // T_x is intended to be arma::mat or arma::sp_mat
    template <typename T_loss, typename T_x>
    class AbclassGroup : public Abclass<T_loss, T_x>
    {
    protected:
        using Abclass<T_loss, T_x>::dn_obs_;
        using Abclass<T_loss, T_x>::km1_;
        using Abclass<T_loss, T_x>::loss_derivative;

        // define gradient function for j-th predictor
        inline arma::rowvec mm_gradient(const arma::vec& inner,
                                        const unsigned int j) const
        {
            arma::vec inner_grad { loss_derivative(inner) };
            arma::rowvec out { arma::zeros<arma::rowvec>(km1_) };
            for (size_t i {0}; i < n_obs_; ++i) {
                out += control_.obs_weight_[i] * inner_grad[i] * x_(i, j) *
                    vertex_.row(y_[i]);
            }
            return out / dn_obs_;
        }
        // for intercept
        inline arma::rowvec mm_gradient0(const arma::vec& inner) const
        {
            arma::vec inner_grad { loss_derivative(inner) };
            arma::rowvec out { arma::zeros<arma::rowvec>(km1_) };
            for (size_t i {0}; i < n_obs_; ++i) {
                out += control_.obs_weight_[i] * inner_grad[i] *
                    vertex_.row(y_[i]);
            }
            return out / dn_obs_;
        }

        // gradient matrix for beta
        inline arma::mat gradient(const arma::vec& inner) const
        {
            arma::mat out { arma::zeros(p0_, km1_) };
            arma::vec inner_grad { loss_derivative(inner) };
            for (size_t j {0}; j < p0_; ++j) {
                arma::rowvec tmp { arma::zeros<arma::rowvec>(km1_) };
                for (size_t i {0}; i < n_obs_; ++i) {
                    tmp += control_.obs_weight_[i] * inner_grad[i] * x_(i, j) *
                        vertex_.row(y_[i]);
                }
                out.row(j) = tmp;
            }
            return out / dn_obs_;
        }

        inline arma::vec gen_group_weight(
            const arma::vec& group_weight = arma::vec()
            ) const
        {
            if (group_weight.n_elem < p0_) {
                arma::vec out { arma::ones(p0_) };
                if (group_weight.is_empty()) {
                    return out;
                }
            } else if (group_weight.n_elem == p0_) {
                if (arma::any(group_weight < 0.0)) {
                    throw std::range_error(
                        "The 'group_weight' cannot be negative.");
                }
                return group_weight;
            }
            // else
            throw std::range_error("Incorrect length of the 'group_weight'.");
        }


    public:
        // inherit constructors
        using Abclass<T_loss, T_x>::Abclass;
        using Abclass<T_loss, T_x>::control_;
        using Abclass<T_loss, T_x>::n_obs_;
        using Abclass<T_loss, T_x>::p0_;
        using Abclass<T_loss, T_x>::x_;
        using Abclass<T_loss, T_x>::y_;
        using Abclass<T_loss, T_x>::vertex_;

        // regularization
        // the "big" enough lambda => zero coef
        double lambda_max_;
        // did user specified a customized lambda sequence?
        bool custom_lambda_ = false;

        // estimates
        arma::cube coef_;         // p1_ by km1_

        // cache
        unsigned int num_iter_;   // number of GMD cycles till convergence

        // for a sequence of lambda's
        virtual void fit() = 0;

        // setter for group weights
        inline void set_group_weight(
            const arma::vec& group_weight = arma::vec()
            )
        {
            control_.group_weight_ = gen_group_weight(group_weight);
        }

    };

}

#endif /* ABCLASS_ABCLASS_GROUP_H */
