#ifndef ABCLASS_LUM_NET_H
#define ABCLASS_LUM_NET_H

#include <RcppArmadillo.h>
#include "AbclassNet.h"
#include "utils.h"

namespace abclass
{
    // define class for inputs and outputs
    class LumNet : public AbclassNet
    {
    private:
        // cache
        double lum_cp1_;        // c + 1
        double lum_c_cp1_;      // c / (c + 1)
        double lum_cma_;        // c - a
        double lum_ap1_;        // a + 1
        double lum_a_ap1_;       // a ^ (a + 1)

    protected:

        double lum_c_ = 0.0;    // c
        double lum_a_ = 1.0;    // a

        // set CMD lowerbound
        inline void set_cmd_lowerbound() override
        {
            double tmp { lum_ap1_ / lum_a_ * lum_cp1_ };
            if (standardize_) {
                cmd_lowerbound_ = arma::ones<arma::rowvec>(p1_);
                cmd_lowerbound_ *= tmp * arma::mean(obs_weight_);
            } else {
                arma::mat sqx { arma::square(x_) };
                sqx.each_col() %= obs_weight_;
                cmd_lowerbound_ = tmp * arma::sum(sqx, 0) / dn_obs_;
            }
        }

        // objective function without regularization
        inline double objective0(const arma::vec& inner) const override
        {
            arma::vec tmp { arma::zeros(inner.n_elem) };
            for (size_t i {0}; i < inner.n_elem; ++i) {
                if (inner[i] < lum_c_cp1_) {
                    tmp[i] = 1.0 - inner[i];
                } else {
                    tmp[i] = std::pow(lum_a_ / (lum_cp1_ * inner[i] - lum_cma_),
                                      lum_a_) / lum_cp1_;
                }
            }
            return arma::mean(obs_weight_ % tmp);
        }

        // the first derivative of the loss function
        inline arma::vec neg_loss_derivative(const arma::vec& u) const override
        {
            arma::vec out { arma::ones(u.n_elem) };
            for (size_t i {0}; i < u.n_elem; ++i) {
                if (u[i] > lum_c_cp1_) {
                    out[i] = lum_a_ap1_ /
                        std::pow(lum_cp1_ * u[i] - lum_cma_, lum_ap1_);
                }
            }
            return out;
        }

    public:

        // inherit constructors
        using AbclassNet::AbclassNet;

        //! @param x The design matrix without an intercept term.
        //! @param y The category index vector.
        LumNet(const arma::mat& x,
               const arma::uvec& y,
               const double lum_a = 1.0,
               const double lum_c = 0.0,
               const bool intercept = true,
               const bool standardize = true,
               const arma::vec& weight = arma::vec()) :
            AbclassNet(x, y, intercept, standardize, weight)
        {
            if (is_le(lum_a, 0.0)) {
                throw std::range_error("The LUM 'a' must be positive.");
            }
            lum_a_ = lum_a;
            lum_ap1_ = lum_a_ + 1.0;
            lum_a_ap1_ = std::pow(lum_a_, lum_ap1_);
            if (is_lt(lum_c, 0.0)) {
                throw std::range_error("The LUM 'c' cannot be negative.");
            }
            lum_cp1_ = lum_c + 1.0;
            lum_c_cp1_ = lum_c_ / lum_cp1_;
            lum_cma_ = lum_c_ - lum_a_;
            // set the CMD lowerbound (which needs to be done only once)
            // set_cmd_lowerbound();
        }


    };                          // end of class

}

#endif
