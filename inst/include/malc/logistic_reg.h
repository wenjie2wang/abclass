#ifndef LOGISTIC_REG_H
#define LOGISTIC_REG_H

#include <RcppArmadillo.h>
#include "utils.h"

namespace Malc {

    // define class for inputs and outputs
    class LogisticReg
    {
    protected:
        unsigned int n_obs_;    // number of observations
        unsigned int k_;        // number of categories
        unsigned int km1_;      // k - 1
        unsigned int p0_;       // number of predictors without intercept
        unsigned int p1_;       // number of predictors (with intercept)
        arma::mat x_;           // (standardized) x_: n by p (with intercept)
        arma::uvec y_;          // y vector ranging in {1, ..., k}
        arma::mat vertex_;      // unique vertex: k by (k - 1)
        arma::mat vertex_mat_;  // vertex matrix: n by (k - 1)
        // arma::mat offset_;      // offset term: n by (k - 1)
        bool intercept_;
        unsigned int int_intercept_;
        bool standardize_;      // is x_ standardized (column-wise)
        arma::rowvec x_center_; // the column center of x_
        arma::rowvec x_scale_;  // the column scale of x_
        // for regularized coordinate majorization descent
        arma::rowvec cmd_lowerbound_; // 1 by p
        arma::mat coef0_;        // coef before rescaling: p by (k - 1)

    public:
        arma::mat l1_penalty_factor_; // adaptive weights for lasso penalty
        double l1_lambda_max_;        // the "big enough" lambda => zero coef
        // for a sinle l1_lambda and l2_lambda
        double alpha_;           // tuning parameter
        double l1_lambda_;       // tuning parameter for lasso penalty
        double l2_lambda_;       // tuning parameter for ridge penalty
        arma::mat coef_;         // coef (rescaled for origin x_)
        arma::mat en_coef_;      // (rescaled) elastic net estimates
        arma::mat pred_mat_;     // prediction matrix: n by k
        arma::mat prob_mat_;     // class conditional probability matrix: n by k
        double pmin_ = 1e-5;

        // for a lambda sequence
        // arma::vec lambda_vec_;   // lambda sequence
        // arma::mat coef_mat_;     // coef matrix (rescaled for origin x_)
        // arma::mat en_coef_mat_;  // elastic net estimates
        // arma::uvec coef_df_vec_; // coef df vector

        // default constructor
        LogisticReg() {}

        //! @param x_ The design matrix without an intercept term.
        //! @param y The category index vector.
        LogisticReg(const arma::mat& x,
                    const arma::uvec& y,
                    const bool intercept = true,
                    const bool standardize = true) :
            x_ (x),
            y_ (y),
            intercept_ (intercept),
            standardize_ (standardize)
        {
            int_intercept_ = static_cast<unsigned int>(intercept_);
            k_ = arma::max(y_);
            km1_ = k_ - 1;
            n_obs_ = x_.n_rows;
            p0_ = x_.n_cols;
            p1_ = p0_ + int_intercept_;
            if (standardize_) {
                if (intercept_) {
                    x_center_ = arma::mean(x_);
                } else {
                    x_center_ = arma::zeros<arma::rowvec>(x_.n_cols);
                }
                x_scale_ = arma::stddev(x_, 1);
                for (size_t j {0}; j < x_.n_cols; ++j) {
                    if (x_scale_(j) > 0) {
                        x_.col(j) = (x_.col(j) - x_center_(j)) / x_scale_(j);
                    } else {
                        throw std::range_error(
                            "The design 'x_' contains constant column."
                            );
                    }
                }
            }
            if (intercept_) {
                x_ = arma::join_horiz(arma::ones(x_.n_rows), x_);
            }
            // set vertex matrix
            set_vertex_matrix(y_, k_);
        }
        // prepare the vertex matrix for observed y {1, ..., k}
        inline void set_vertex_matrix(const arma::uvec& y,
                                      const unsigned int k)
        {
            Simplex sim { k };
            vertex_ = sim.get_vertex();
            vertex_mat_ = arma::zeros(y.n_elem, k - 1);
            for (size_t i { 0 }; i < y.n_elem; ++i) {
                vertex_mat_.row(i) = vertex_.row(y(i) - 1);
            }
        }
        // transfer coef for standardized data to coef for non-standardized data
        inline void rescale_coef()
        {
            coef_ = coef0_;
            if (standardize_) {
                if (intercept_) {
                    // for each columns
                    for (size_t k { 0 }; k < km1_; ++k) {
                        arma::vec coef_k { coef0_.col(k) };
                        coef_(0, k) = coef0_(0, k) -
                            arma::as_scalar((x_center_ / x_scale_) *
                                            coef_k.tail_rows(p0_));
                        for (size_t l { 1 }; l < coef_k.n_elem; ++l) {
                            coef_(l, k) = coef_k(l) / x_scale_(l - 1);
                        }
                    }
                } else {
                    for (size_t k { 0 }; k < km1_; ++k) {
                        for (size_t l { 0 }; l < p0_; ++l) {
                            coef_(l, k) /= x_scale_(l);
                        }
                    }
                }
            }
        }
        // compute cov lowerbound used in regularied model
        inline void set_cmd_lowerbound()
        {
            cmd_lowerbound_ = arma::sum(arma::square(x_), 0) / (4 * x_.n_rows);
        }
        // objective function without regularization
        inline double objective0(const arma::vec& inner) const
        {
            return arma::sum(1.0 / (1.0 + arma::exp(inner))) / n_obs_;
        }
        inline double regularization(const arma::mat& beta) const
        {
            if (intercept_) {
                arma::mat beta0int { beta.tail_rows(p0_) };
                return l1_lambda_ *
                    arma::accu(beta0int %
                               arma::abs(l1_penalty_factor_.tail_rows(p0_))) +
                    l2_lambda_ * arma::accu(arma::square(beta0int));
            }
            return l1_lambda_ *
                arma::accu(arma::abs(l1_penalty_factor_ % beta)) +
                l2_lambda_ * arma::accu(arma::square(beta));
        }
        // objective function with regularization
        inline double objective(const arma::vec& inner,
                                const arma::mat& beta) const
        {
            return objective0(inner) + regularization(beta);
        }
        // the first derivative of the loss function
        inline arma::vec loss_derivative(const arma::vec& u) const
        {
            return - 1.0 / (1.0 + arma::exp(u));
        }
        inline double loss_derivative(const double u) const
        {
            return - 1.0 / (1.0 + std::exp(u));
        }
        // define gradient function at k-th dimension
        inline double cmd_gradient(const arma::vec& inner,
                                   const unsigned int l,
                                   const unsigned int j) const
        {
            double out { 0.0 };
            for (size_t i { 0 }; i < inner.n_elem; ++i) {
                double p_est { - loss_derivative(inner(i)) };
                if (p_est < pmin_){
                    p_est = pmin_;
                } else if (p_est > 1 - pmin_) {
                    p_est = 1 - pmin_;
                }
                out += vertex_mat_(i, j) * x_(i, l) * p_est;
            }
            return - out / n_obs_;
        }
        // gradient matrix
        inline arma::mat gradient(const arma::vec& inner) const
        {
            arma::mat out { arma::zeros(p1_, km1_) };
            for (size_t j { 0 }; j < km1_; ++j) {
                for (size_t l { 0 }; l < p1_; ++l) {
                    out(l, j) = cmd_gradient(inner, l, j);
                }
            }
            return out;
        }
        // inner products
        inline void compute_pred_mat()
        {
            arma::mat x_beta { x_ * coef0_ };
            pred_mat_ = x_beta * vertex_.t();
        }
        // class conditional probability
        inline void compute_prob_mat()
        {
            prob_mat_ = arma::zeros(n_obs_, k_);
            for (size_t j { 0 }; j < k_; ++j) {
                prob_mat_.col(j)  = 1 / loss_derivative(pred_mat_.col(j));
            }
            for (size_t i { 0 }; i < n_obs_; ++i) {
                prob_mat_.row(i) /= arma::sum(prob_mat_.row(i));
            }
        }
        // run one cycle of coordinate descent over a given active set
        inline void run_one_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::umat& is_active,
                                         const double l1_lambda,
                                         const double l2_lambda,
                                         const arma::mat& l1_penalty_factor,
                                         const bool update_active,
                                         const bool early_stop,
                                         const bool verbose);
        // run a complete cycle of CMD for a given active set and given lambda's
        inline void run_cmd_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::umat& is_active,
                                         const double l1_lambda,
                                         const double l2_lambda,
                                         const arma::mat& l1_penatly_factor,
                                         const bool varying_active_set,
                                         const unsigned int max_iter,
                                         const double rel_tol,
                                         const bool early_stop,
                                         const bool verbose);

        // for a perticular lambda
        inline void elastic_net(const double l1_lambda,
                                const double l2_lambda,
                                const arma::mat& l1_penalty_factor,
                                const arma::mat& start,
                                const unsigned int max_iter,
                                const double rel_tol,
                                const double pmin,
                                const bool early_stop,
                                const bool verbose);


    };                          // end of class

    inline void LogisticReg::run_one_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
        const arma::mat& l1_penalty_factor,
        const bool update_active = false,
        const bool early_stop = false,
        const bool verbose = false
        )
    {
        double dlj { 0.0 };
        arma::mat beta_old;
        arma::vec inner_old;
        if (early_stop || verbose) {
            beta_old = beta;
            inner_old = inner;
        }
        for (size_t j { 0 }; j < km1_; ++j) {
            for (size_t l { 0 }; l < p1_; ++l) {
                if (is_active(l, j) > 0) {
                    dlj = cmd_gradient(inner, l, j);
                    double tmp { beta(l, j) };
                    // update beta
                    beta(l, j) = soft_threshold(
                        cmd_lowerbound_(l) * beta(l, j) - dlj,
                        l1_penalty_factor(l, j) * l1_lambda
                        ) / (cmd_lowerbound_(l) + 2 * l2_lambda *
                             static_cast<double>(l >= int_intercept_));
                    inner += (beta(l, j) - tmp) *
                        (x_.col(l) % vertex_mat_.col(j));
                    if (update_active) {
                        // check if it has been shrinkaged to zero
                        if (isAlmostEqual(beta(l, j), 0)) {
                            is_active(l, j) = 0;
                        } else {
                            is_active(l, j) = 1;
                        }
                    }
                }
            }
        }
        // if early stop, check improvement
        if (early_stop || verbose) {
            double ell_old { objective(inner_old, beta_old) };
            double ell_new { objective(inner, beta) };
            if (verbose) {
                Rcpp::Rcout << "The objective function changed\n";
                Rprintf("  from %15.15f\n", ell_old);
                Rprintf("    to %15.15f\n", ell_new);
            }
            if (early_stop && ell_new > ell_old) {
                if (verbose) {
                    Rcpp::Rcout << "Warning: "
                                << "the objective function somehow increased\n";
                    Rcpp::Rcout << "\nEarly stopped the CMD iterations "
                                << "with estimates from the last step"
                                << std::endl;
                }
                beta = beta_old;
                inner = inner_old;
            }
        }
    }

    inline void LogisticReg::run_cmd_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
        const arma::mat& l1_penalty_factor,
        const bool varying_active_set,
        const unsigned int max_iter,
        const double rel_tol = false,
        const bool early_stop = false,
        const bool verbose = false
        )
    {
        size_t i {0};
        arma::mat beta0 { beta };
        arma::umat is_active_stored { is_active };

        // use active-set if p > n ("helps when p >> n")
        if (varying_active_set) {
            arma::umat is_active_new { is_active };
            size_t ii {0};
            while (i < max_iter) {
                // cycles over the active set
                while (ii < max_iter) {
                    run_one_active_cycle(beta, inner, is_active_stored,
                                         l1_lambda, l2_lambda,
                                         l1_penalty_factor, true,
                                         early_stop, verbose);
                    if (rel_l1_norm(beta, beta0) < rel_tol) {
                        break;
                    }
                    beta0 = beta;
                    ii++;
                }
                // run a full cycle over the converged beta
                run_one_active_cycle(beta, inner, is_active_new,
                                     l1_lambda, l2_lambda,
                                     l1_penalty_factor, true,
                                     early_stop, verbose);
                // check two active sets coincide
                if (l1_norm(is_active_new - is_active_stored)) {
                    // if different, repeat this process
                    ii = 0;
                    i++;
                } else {
                    break;
                }
            }
        } else {
            // regular coordinate descent
            while (i < max_iter) {
                run_one_active_cycle(beta, inner, is_active_stored,
                                     l1_lambda, l2_lambda,
                                     l1_penalty_factor, false,
                                     early_stop, verbose);
                if (rel_l1_norm(beta, beta0) < rel_tol) {
                    break;
                }
                beta0 = beta;
                i++;
            }
        }
    }

    // for particular lambda's
    // lambda_1 * factor * lasso + lambda_2 * ridge
    inline void LogisticReg::elastic_net(
        const double l1_lambda,
        const double l2_lambda,
        const arma::mat& l1_penalty_factor,
        const arma::mat& start,
        const unsigned int max_iter,
        const double rel_tol,
        const double pmin,
        const bool early_stop,
        const bool verbose
        )
    {
        pmin_ = pmin;
        l1_lambda_ = l1_lambda;
        l2_lambda_ = l2_lambda;
        // set the CMD lowerbound (which needs to be done only once)
        set_cmd_lowerbound();
        // set penalty terms
        l1_penalty_factor_ = arma::ones(p0_, km1_);
        if (arma::size(l1_penalty_factor) == arma::size(l1_penalty_factor_)) {
            l1_penalty_factor_ = l1_penalty_factor * p0_ /
                arma::accu(l1_penalty_factor);
        }
        arma::vec inner { arma::zeros(n_obs_) };
        arma::mat beta { arma::zeros(p1_, km1_) };
        arma::mat grad_zero { arma::abs(gradient(inner)) };
        arma::mat grad_beta { grad_zero }, strong_rhs { grad_beta };
        // large enough lambda for all-zero coef (except intercept terms)
        // excluding variable with zero penalty factor
        arma::uvec active_l1_penalty { arma::find(l1_penalty_factor_ > 0) };
        grad_zero = grad_zero.tail_rows(p0_);
        l1_lambda_max_ = arma::max(grad_zero.elem(active_l1_penalty) /
                                   l1_penalty_factor_.elem(active_l1_penalty));
        if (intercept_) {
            l1_penalty_factor_ = arma::join_vert(
                arma::zeros<arma::rowvec>(km1_), l1_penalty_factor_);
        }
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::umat is_active_strong { arma::zeros<arma::umat>(p1_, km1_) };
        if (intercept_) {
            // only need to estimate intercept
            is_active_strong.row(0) = arma::ones<arma::umat>(1, km1_);
            run_cmd_active_cycle(beta, inner, is_active_strong,
                                 l1_lambda_max_, l2_lambda_, l1_penalty_factor_,
                                 false, max_iter, rel_tol, early_stop, verbose);
        }
        coef0_ = beta;
        rescale_coef();

        // early exit for lambda greater than lambda_max
        if (l1_lambda >= l1_lambda_max_) {
            en_coef_ = coef_;
            compute_pred_mat();
            compute_prob_mat();
            return;
        }

        // use the input start if correctly specified
        if (start.size() == x_.size()) {
            beta = start;
        }

        // update active set by strong rule
        grad_beta = arma::abs(gradient(inner));
        strong_rhs = (2 * l1_lambda_ - l1_lambda_max_) * l1_penalty_factor_;

        for (size_t j { 0 }; j < km1_; ++j) {
            for (size_t l { int_intercept_ }; l < p1_; ++l) {
                if (grad_beta(l, j) >= strong_rhs(l, j)) {
                    is_active_strong(l, j) = 1;
                } else {
                    beta(l, j) = 0;
                }
            }
        }

        arma::umat is_active_strong_new { is_active_strong };
        // optim with varying active set when p > n
        bool varying_active_set { false };
        if (p1_ > n_obs_ || p1_ > 50) {
            varying_active_set = true;
        }

        bool kkt_failed { true };
        strong_rhs = l1_lambda_ * l1_penalty_factor_;
        // eventually, strong rule will guess correctly
        while (kkt_failed) {
            // update beta
            run_cmd_active_cycle(beta, inner, is_active_strong,
                                 l1_lambda_, l2_lambda_, l1_penalty_factor_,
                                 varying_active_set,
                                 max_iter, rel_tol, early_stop, verbose);
            // check kkt condition
            for (size_t j { 0 }; j < km1_; ++j) {
                for (size_t l { int_intercept_ }; l < p1_; ++l) {
                    if (is_active_strong(l, j)) {
                        continue;
                    }
                    if (std::abs(cmd_gradient(inner, l, j)) >
                        strong_rhs(l, j)) {
                        // update active set
                        is_active_strong_new(l, j) = 1;
                    }
                }
            }
            if (l1_norm(is_active_strong - is_active_strong_new)) {
                is_active_strong = is_active_strong_new;
            } else {
                kkt_failed = false;
            }
        }
        // compute elastic net estimates, then rescale them back
        coef0_ = (1 + l2_lambda) * beta;
        rescale_coef();
        en_coef_ = coef_;
        // overwrite the naive elastic net estimate
        coef0_ = beta;
        rescale_coef();
        // compute probability matrix
        compute_pred_mat();
        compute_prob_mat();
    }

}  // Malc

#endif /* LOGISTIC_REG_H */
