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
        inline virtual double dloss_du(const double u) const = 0;

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
        inline arma::vec dloss_du(const arma::vec& u,
                                  const arma::vec& obs_weight) const
        {
            arma::vec out { arma::zeros(u.n_elem) };
            for (size_t i {0}; i < out.n_elem; ++i) {
                out[i] = obs_weight(i) * dloss_du(u[i]);
            }
            return out;
        }

        inline arma::vec dloss_du(const arma::vec& u) const
        {
            arma::vec out { arma::zeros(u.n_elem) };
            for (size_t i {0}; i < out.n_elem; ++i) {
                out[i] = dloss_du(u[i]);
            }
            return out;
        }

        // wrappers for Abclass
        // a margin-based loss that depends on inner product
        template <typename T_x>
        inline double loss(const Simplex2<T_x>& data,
                           const arma::vec& obs_weight) const
        {
            return loss(data.iter_inner_, obs_weight);
        }

        // gradient of loss wrt the (K-1) decision functions
        template <typename T_x>
        inline arma::mat dloss_df(const Simplex2<T_x>& data,
                                  const arma::vec& obs_weight) const
        {
            arma::mat out { data.ex_vertex_ };
            arma::vec dloss_u { dloss_du(data.iter_inner_, obs_weight) };
            out.each_col() %= dloss_u;
            return out;
        }

        // gradient of loss wrt the k-th decision function
        template <typename T_x>
        inline arma::vec dloss_df(const Simplex2<T_x>& data,
                                  const arma::vec& obs_weight,
                                  const unsigned int k) const
        {
            arma::vec out { data.ex_vertex_.col(k) };
            arma::vec dloss_u { dloss_du(data.iter_inner_, obs_weight) };
            out %= dloss_u;
            return out;
        }

        // for linear learning
        // gradient wrt beta_g.
        template <typename T_x>
        inline arma::mat dloss_dbeta(Simplex2<T_x>& data,
                                     const arma::vec& obs_weight,
                                     const unsigned int g) const
        {
            arma::mat vxg { data.ex_vertex_ };
            vxg.each_col() %= data.x_.col(g);
            // cache it in data for updating pred_f and inner
            data.iter_v_xg_ = vxg;
            vxg.each_col() %= dloss_du(data.iter_inner_, obs_weight);
            return vxg;
        }

        // gradient wrt beta_gk
        template <typename T_x>
        inline arma::vec dloss_dbeta(Simplex2<T_x>& data,
                                     const arma::vec& obs_weight,
                                     const unsigned int g,
                                     const unsigned int k) const
        {
            arma::vec vkxg { data.ex_vertex_.col(k) % data.x_.col(g) };
            // cache it in data for updating pred_f and inner
            data.iter_vk_xg_ = vkxg;
            return dloss_du(data.iter_inner_, obs_weight) % vkxg;
        }

        // probability score for the decision function of the k-th class
        inline arma::vec prob_score_k(const arma::vec& pred_k) const
        {
            return 1.0 / dloss_du(pred_k);
        }

    };

}  // abclass


#endif /* ABCLASS_MARGIN_LOSS_H */
