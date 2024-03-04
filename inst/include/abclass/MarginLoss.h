#ifndef ABCLASS_MARGIN_LOSS_H
#define ABCLASS_MARGIN_LOSS_H

#include <RcppArmadillo.h>
#include "Simplex.h"

namespace abclass
{
    // base class for margin-based loss functions
    class MarginLoss
    {
    public:
        MarginLoss() {}

        // pure virtual
        inline virtual double loss(const double u) const = 0;
        inline virtual double dloss(const double u) const = 0;

        // loss function with observational weights
        inline double loss(const arma::vec& u,
                           const arma::vec& obs_weight) const
        {
            double res { 0.0 };
            for (size_t i {0}; i < u.n_elem; ++i) {
                res += obs_weight(i) * loss(u[i]);
            }
            return res;
        }
        // the first derivative with observational weights
        inline arma::vec dloss(const arma::vec& u,
                               const arma::vec& obs_weight) const
        {
            arma::vec out { arma::zeros(u.n_elem) };
            for (size_t i {0}; i < out.n_elem; ++i) {
                out[i] = obs_weight(i) * dloss(u[i]);
            }
            return out;
        }

        // wrappers for Abclass
        // a margin-based loss that depends on inner product
        inline double loss(const Simplex2& resp,
                           const arma::vec& obs_weight) const
        {
            return loss(resp.iter_inner_, obs_weight);
        }
        inline arma::vec dloss(const Simplex2& resp,
                               const arma::vec& obs_weight) const
        {
            return dloss(resp.iter_inner_, obs_weight);
        }

    };

}  // abclass


#endif /* ABCLASS_MARGIN_LOSS_H */
