#ifndef ABCLASS_BOOST_NET_H
#define ABCLASS_BOOST_NET_H

#include <RcppArmadillo.h>
#include <stdexcept>
#include "AbclassNet.h"
#include "utils.h"

namespace abclass
{
    // define class for inputs and outputs
    class BoostNet : public AbclassNet
    {
    private:
        // cache
        double exp_inner_max_;

    protected:

        double inner_min_ = - 3.0;

        // set CMD lowerbound
        inline void set_cmd_lowerbound() override
        {
            if (standardize_) {
                cmd_lowerbound_ = arma::ones<arma::rowvec>(p1_);
                cmd_lowerbound_ *= exp_inner_max_ * arma::mean(obs_weight_);
            } else {
                arma::mat sqx { arma::square(x_) };
                sqx.each_col() %= obs_weight_;
                cmd_lowerbound_ = exp_inner_max_ * arma::sum(sqx, 0) / dn_obs_;
            }
        }

        // objective function without regularization
        inline double objective0(const arma::vec& inner) const override
        {
            arma::vec tmp { arma::zeros(inner.n_elem) };
            double tmp1 { 1 - inner_min_ };
            for (size_t i {0}; i < inner.n_elem; ++i) {
                if (inner[i] < inner_min_) {
                    tmp[i] = (tmp1 - inner[i]) * exp_inner_max_;
                } else {
                    tmp[i] = std::exp(- inner[i]);
                }
            }
            return arma::mean(obs_weight_ % tmp);
        }

        // the first derivative of the loss function
        inline arma::vec neg_loss_derivative(const arma::vec& u) const override
        {
            arma::vec out { arma::zeros(u.n_elem) };
            for (size_t i {0}; i < u.n_elem; ++i) {
                if (u[i] < inner_min_) {
                    out[i] = exp_inner_max_;
                } else {
                    out[i] = std::exp(- u[i]);
                }
            }
            return out;
        }

    public:

        // inherit constructors
        using AbclassNet::AbclassNet;

        //! @param x The design matrix without an intercept term.
        //! @param y The category index vector.
        BoostNet(const arma::mat& x,
                 const arma::uvec& y,
                 const double inner_min = - 3.0,
                 const bool intercept = true,
                 const bool standardize = true,
                 const arma::vec& weight = arma::vec()) :
            AbclassNet(x, y, intercept, standardize, weight)
        {
            if (is_gt(inner_min, 0.0)) {
                throw std::range_error("The 'inner_min' cannot be positive.");
            }
            inner_min_ = inner_min;
            exp_inner_max_ = std::exp(- inner_min_);
            // set the CMD lowerbound (which needs to be done only once)
            set_cmd_lowerbound();
        }


    };                          // end of class

}

#endif
