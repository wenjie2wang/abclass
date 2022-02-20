#ifndef ABCLASS_LOGISTIC_NET_H
#define ABCLASS_LOGISTIC_NET_H

#include <RcppArmadillo.h>
#include "AbclassNet.h"

namespace abclass
{
    // define class for inputs and outputs
    class LogisticNet : public AbclassNet
    {
    protected:

        // set CMD lowerbound
        inline void set_cmd_lowerbound() override
        {
            arma::mat sqx { arma::square(x_) };
            sqx.each_col() %= obs_weight_;
            cmd_lowerbound_ = arma::sum(sqx, 0) / (4.0 * dn_obs_);
        }

        // objective function without regularization
        inline double objective0(const arma::vec& inner) const override
        {
            return arma::sum(arma::log(1.0 + arma::exp(- inner))) / dn_obs_;
        }

        // the first derivative of the loss function
        inline arma::vec neg_loss_derivative(const arma::vec& u) const override
        {
            return 1.0 / (1.0 + arma::exp(u));
        }

    public:

        //! @param x The design matrix without an intercept term.
        //! @param y The category index vector.
        LogisticNet(const arma::mat& x,
                    const arma::uvec& y,
                    const bool intercept = true,
                    const bool standardize = true,
                    const arma::vec& weight = arma::vec()) :
            AbclassNet(x, y, intercept, standardize, weight)
        {
            // set the CMD lowerbound (which needs to be done only once)
            set_cmd_lowerbound();
        }


    };                          // end of class

}

#endif
