#ifndef ABCLASS_LOGISTIC_H
#define ABCLASS_LOGISTIC_H

#include <RcppArmadillo.h>
#include "utils.h"

namespace abclass
{

    class Logistic
    {
    public:
        Logistic();

        // loss function
        inline double loss(const arma::vec& u,
                           const arma::vec& obs_weight) const
        {
            return arma::mean(obs_weight % arma::log(1.0 + arma::exp(- u)));
        }

        // the first derivative of the loss function
        inline arma::vec dloss(const arma::vec& u) const
        {
            arma::vec out { arma::zeros(u.n_elem) };
            for (size_t i {0}; i < out.n_elem; ++i) {
                out[i] = - 1.0 / (1.0 + std::exp(u[i]));
            }
            return out;
            // return - 1.0 / (1.0 + arma::exp(u));
        }

        // MM lowerbound
        template <typename T>
        inline arma::rowvec mm_lowerbound(const T& x,
                                          const arma::vec& obs_weight)
        {
            T sqx { arma::square(x) };
            double dn_obs { static_cast<double>(x.n_rows) };
            return obs_weight.t() * sqx / (4.0 * dn_obs);

        }
        // for the intercept
        inline double mm_lowerbound(const double dn_obs,
                                    const arma::vec& obs_weight)
        {
            return arma::accu(obs_weight) / (4.0 * dn_obs);
        }

    };

}  // abclass

#endif /* ABCLASS_LOGISTIC_H */
