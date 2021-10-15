#ifndef ABCLASS_LOGISTIC_NET_H
#define ABCLASS_LOGISTIC_NET_H

#include <utility>
#include <vector>
#include <RcppArmadillo.h>
#include "utils.h"

namespace Abclass {

    // define class for inputs and outputs
    class LogisticNet
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

        // for regularized coordinate majorization descent
        arma::rowvec cmd_lowerbound_; // 1 by p

        // for one lambda
        arma::mat coef0_;         // coef (not scaled for the origin x_)

    public:
        // common
        double l1_lambda_max_;        // the "big enough" lambda => zero coef
        double alpha_;
        double pmin_ = 1e-5;
        bool path_ = true;

        // for a sinle l1_lambda and l2_lambda
        double lambda_;
        double l1_lambda_;       // tuning parameter for lasso penalty
        double l2_lambda_;       // tuning parameter for ridge penalty
        arma::mat coef_;

        // for a lambda sequence
        arma::vec lambda_path_;   // lambda sequence
        arma::cube coef_path_;

        // default constructor
        LogisticNet() {}

        //! @param x The design matrix without an intercept term.
        //! @param y The category index vector.
        LogisticNet(const arma::mat& x,
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
                obs_weight_ = weight / arma::sum(weight) * n_obs_;
            }
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
                        x_.col(j) = arma::zeros(x_.n_rows);
                        // make scale(j) nonzero for rescaling
                        x_scale_(j) = - 1.0;
                    }
                }
            }
            if (intercept_) {
                x_ = arma::join_horiz(arma::ones(x_.n_rows), x_);
            }
            // set vertex matrix
            set_vertex_matrix(k_);
            // set the CMD lowerbound (which needs to be done only once)
            set_cmd_lowerbound();
        }
        // prepare the vertex matrix for observed y {1, ..., k}
        inline void set_vertex_matrix(const unsigned int k)
        {
            Simplex sim { k };
            vertex_ = sim.get_vertex();
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
        inline void rescale_coef()
        {
            coef_ = rescale_coef(coef0_);
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

        // compute cov lowerbound used in regularied model
        inline void set_cmd_lowerbound()
        {
            arma::mat sqx { arma::square(x_) };
            sqx.each_col() %= obs_weight_;
            cmd_lowerbound_ = arma::sum(sqx, 0) / (4.0 * n_obs_);
        }
        // objective function without regularization
        inline double objective0(const arma::vec& inner) const
        {
            return arma::sum(arma::log(1.0 + arma::exp(- inner))) / n_obs_;
        }

        inline double regularization(const arma::mat& beta) const
        {
            if (intercept_) {
                arma::mat beta0int { beta.tail_rows(p0_) };
                return l1_lambda_ * arma::accu(beta0int) +
                    l2_lambda_ * arma::accu(arma::square(beta0int));
            }
            return l1_lambda_ * arma::accu(arma::abs(beta)) +
                l2_lambda_ * arma::accu(arma::square(beta));
        }
        // objective function with regularization
        inline double objective(const arma::vec& inner,
                                const arma::mat& beta) const
        {
            return objective0(inner) + regularization(beta);
        }

        // the first derivative of the loss function
        inline arma::vec neg_loss_derivative(const arma::vec& u) const
        {
            return 1.0 / (1.0 + arma::exp(u));
        }
        inline double neg_loss_derivative(const double u) const
        {
            return 1.0 / (1.0 + std::exp(u));
        }
        inline arma::vec loss_derivative(const arma::vec& u) const
        {
            return - neg_loss_derivative(u);
        }
        inline double loss_derivative(const double u) const
        {
            return - neg_loss_derivative(u);
        }

        // define gradient function at k-th dimension
        inline double cmd_gradient(const arma::vec& inner,
                                   const unsigned int l,
                                   const unsigned int j) const
        {
            double out { 0.0 };
            for (size_t i { 0 }; i < inner.n_elem; ++i) {
                double p_est { neg_loss_derivative(inner(i)) };
                if (p_est < pmin_) {
                    p_est = pmin_;
                } else if (p_est > 1 - pmin_) {
                    p_est = 1 - pmin_;
                }
                out += obs_weight_(i) * vertex_(y_(i), j) * x_(i, l) * p_est;
            }
            return - out / static_cast<double>(n_obs_);
        }
        // gradient matrix
        inline arma::mat gradient(const arma::vec& inner) const
        {
            arma::mat out { arma::zeros(p1_, km1_) };
            arma::vec p_vec { neg_loss_derivative(inner) };
            for (size_t i { 0 }; i < p_vec.n_elem; ++i) {
                if (p_vec(i) < pmin_) {
                    p_vec(i) = pmin_;
                } else if (p_vec(i) > 1 - pmin_) {
                    p_vec(i) = 1 - pmin_;
                }
            }
            arma::mat::row_col_iterator it { out.begin_row_col() };
            arma::mat::row_col_iterator it_end { out.end_row_col() };
            for (; it != it_end; ++it) {
                double tmp { 0.0 };
                for (size_t i { 0 }; i < inner.n_elem; ++i) {
                    tmp += obs_weight_(i) * vertex_(y_(i), it.col()) *
                        x_(i, it.row()) * p_vec(i);
                }
                *it = - tmp / dn_obs_;
            }
            return out;
        }
        // class conditional probability
        inline arma::mat compute_prob_mat(const arma::mat& beta,
                                          const arma::mat& x) const
        {
            arma::mat out { x * beta };
            out *= vertex_.t();
            // for (size_t j { 0 }; j < k_; ++j) {
            //     out.col(j)  = 1 / loss_derivative(out.col(j));
            // }
            out.each_col( [&](arma::vec& a) {
                a = 1.0 / loss_derivative(a);
            });
            arma::vec row_sums { arma::sum(out, 1) };
            // for (size_t j { 0 }; j < k_; ++j) {
            //     out.col(j) /= row_sums;
            // }
            out.each_col( [&row_sums](arma::vec& a) { a /= row_sums; } ) ;
            return out;
        }
        inline arma::mat compute_prob_mat(const arma::mat& beta) const
        {
            return compute_prob_mat(beta, x_);
        }

        // predict categories for the training set
        inline arma::uvec predict_cat(const arma::mat& prob_mat) const
        {
            return arma::index_max(prob_mat, 1);
        }
        // number of incorrect classification
        inline unsigned int miss_number(const arma::mat& beta,
                                        const arma::mat& x,
                                        const arma::uvec& y) const
        {
            arma::mat prob_mat { compute_prob_mat(beta, x) };
            arma::uvec max_idx { predict_cat(prob_mat) };
            return static_cast<double>(arma::sum(max_idx != y));
        }
        // accuracy
        inline double accuracy(const arma::mat& beta,
                               const arma::mat& x,
                               const arma::uvec& y) const
        {
            arma::mat prob_mat { compute_prob_mat(beta, x) };
            arma::uvec max_idx { predict_cat(prob_mat) };
            return static_cast<double>(arma::sum(max_idx == y)) / y.n_elem;
        }

        // run one cycle of coordinate descent over a given active set
        inline void run_one_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::umat& is_active,
                                         const double l1_lambda,
                                         const double l2_lambda,
                                         const bool update_active,
                                         const bool verbose);
        // run complete cycles of CMD for a given active set and given lambda's
        inline void run_cmd_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::umat& is_active,
                                         const double l1_lambda,
                                         const double l2_lambda,
                                         const bool varying_active_set,
                                         const unsigned int max_iter,
                                         const double rel_tol,
                                         const bool verbose);
        // one full cycle for coordinate-descent
        inline void run_one_full_cycle(arma::mat& beta,
                                       arma::vec& inner,
                                       const double l1_lambda,
                                       const double l2_lambda);
        // run full cycles of CMD for given lambda's
        inline void run_cmd_full_cycle(arma::mat& beta,
                                       arma::vec& inner,
                                       const double l1_lambda,
                                       const double l2_lambda,
                                       const unsigned int max_iter,
                                       const double rel_tol);

        // for a perticular lambda
        inline void fit(const double lambda,
                        const double alpha,
                        const arma::mat& start,
                        const unsigned int max_iter,
                        const double rel_tol,
                        const double pmin,
                        const bool verbose);

        // for a sequence of lambda's
        inline void path(const arma::vec& lambda,
                         const double alpha,
                         const unsigned int nlambda,
                         const double lambda_min_ratio,
                         const unsigned int max_iter,
                         const double rel_tol,
                         const double pmin,
                         const bool verbose);

        // getters
        inline arma::vec get_weight() const
        {
            return obs_weight_;
        }

    };                          // end of class

    inline void LogisticNet::run_one_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
        const bool update_active = false,
        const bool verbose = false
        )
    {
        double dlj { 0.0 };
        double ell_verbose;
        if (verbose) {
            Rcpp::Rcout << "\nStarting values of beta:\n";
            Rcpp::Rcout << beta << std::endl;
            Rcpp::Rcout << "\nThe active set of beta:\n";
            Rcpp::Rcout << is_active << std::endl;
            ell_verbose = objective(inner, beta);
        };
        // arma::umat::row_col_iterator it { is_active.begin_row_col() };
        // arma::umat::row_col_iterator it_end { is_active.end_row_col() };
        arma::umat is_active_new { is_active };
        for (size_t j {0}; j < is_active.n_cols; ++j) {
            arma::vec vj { vertex_.col(j) };
            vj = vj.elem(y_);
            for (size_t l {0}; l < is_active.n_rows; ++l) {
                // arma::uword l { it.row() };
                // arma::uword j { it.col() };
                if (is_active(l, j) > 0) {
                    dlj = cmd_gradient(inner, l, j);
                    double tmp { beta(l, j) };
                    // if cmd_lowerbound = 0 and l1_lambda > 0, numer will be 0
                    double numer {
                        soft_threshold(
                            cmd_lowerbound_(l) * beta(l, j) - dlj,
                            l1_lambda * static_cast<double>(l >= int_intercept_)
                            )
                    };
                    // update beta
                    if (isAlmostEqual(numer, 0)) {
                        beta(l, j) = 0;
                    } else {
                        double denom { cmd_lowerbound_(l) + 2 * l2_lambda *
                            static_cast<double>(l >= int_intercept_)
                        };
                        beta(l, j) = numer / denom;
                    }
                    inner += (beta(l, j) - tmp) * (x_.col(l) % vj);
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
        is_active = std::move(is_active_new);
        // if early stop, check improvement
        if (verbose) {
            double ell_old { ell_verbose };
            Rcpp::Rcout << "The objective function changed\n";
            Rprintf("  from %15.15f\n", ell_verbose);
            ell_verbose = objective(inner, beta);
            Rprintf("    to %15.15f\n", ell_verbose);
            if (ell_verbose > ell_old) {
                Rcpp::Rcout << "Warning: "
                            << "the objective function somehow increased\n";
            }
        }
    }

    inline void LogisticNet::run_cmd_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
        const bool varying_active_set,
        const unsigned int max_iter,
        const double rel_tol = false,
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
                                         l1_lambda, l2_lambda, true, verbose);
                    if (rel_diff(beta, beta0) < rel_tol) {
                        break;
                    }
                    beta0 = beta;
                    ii++;
                }
                // run a full cycle over the converged beta
                run_one_active_cycle(beta, inner, is_active_new,
                                     l1_lambda, l2_lambda, true, verbose);
                // check two active sets coincide
                if (l1_norm(is_active_new - is_active_stored) > 0) {
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
                                     l1_lambda, l2_lambda, false, verbose);
                if (rel_diff(beta, beta0) < rel_tol) {
                    break;
                }
                beta0 = beta;
                i++;
            }
        }
    }

    // one full cycle for coordinate-descent
    inline void LogisticNet::run_one_full_cycle(
        arma::mat& beta,
        arma::vec& inner,
        const double l1_lambda,
        const double l2_lambda
        )
    {
        for (size_t j {0}; j < beta.n_cols; ++j) {
            arma::vec vj { vertex_.col(j) };
            vj = vj.elem(y_);
            for (size_t l {0}; l < beta.n_rows; ++l) {
                double dlj { cmd_gradient(inner, l, j) };
                double tmp { beta(l, j) };
                // if cmd_lowerbound = 0 and l1_lambda > 0, numer will be 0
                double numer {
                    soft_threshold(
                        cmd_lowerbound_(l) * beta(l, j) - dlj,
                        l1_lambda * static_cast<double>(l >= int_intercept_)
                        )
                };
                // update beta
                if (isAlmostEqual(numer, 0)) {
                    beta(l, j) = 0;
                } else {
                    double denom { cmd_lowerbound_(l) + 2 * l2_lambda *
                        static_cast<double>(l >= int_intercept_)
                    };
                    beta(l, j) = numer / denom;
                }
                inner += (beta(l, j) - tmp) * (x_.col(l) % vj);
            }
        }
    }
    inline void LogisticNet::run_cmd_full_cycle(
        arma::mat& beta,
        arma::vec& inner,
        const double l1_lambda,
        const double l2_lambda,
        const unsigned int max_iter,
        const double rel_tol = false
        )
    {
        arma::mat beta0 { beta };
        for (size_t i {0}; i < max_iter; ++i) {
            run_one_full_cycle(beta, inner, l1_lambda, l2_lambda);
            if (rel_diff(beta, beta0) < rel_tol) {
                break;
            }
            beta0 = beta;
        }
    }

    // for a particular lambda
    // lambda_1 * lasso + lambda_2 * ridge
    inline void LogisticNet::fit(
        const double lambda,
        const double alpha,
        const arma::mat& start,
        const unsigned int max_iter,
        const double rel_tol,
        const double pmin,
        const bool verbose
        )
    {
        // check alpha
        if ((alpha < 0) || (alpha > 1)) {
            throw std::range_error("The 'alpha' must be between 0 and 1.");
        }
        alpha_ = alpha;
        if ((pmin < 0) || (pmin > 1)) {
            throw std::range_error("The 'pmin' must be between 0 and 1.");
        }
        pmin_ = pmin;
        l1_lambda_ = lambda * alpha;
        l2_lambda_ = 0.5 * lambda * (1 - alpha);
        arma::vec inner { arma::zeros(n_obs_) };
        arma::mat beta { arma::zeros(p1_, km1_) };
        // ridge penalty only
        if (isAlmostEqual(l1_lambda_, 0.0)) {
            // use the input start if correctly specified
            if (start.size() == x_.size()) {
                if (standardize_) {
                    beta = rev_rescale_coef(start);
                } else {
                    beta = start;
                }
            }
            run_cmd_full_cycle(beta, inner, l1_lambda_, l2_lambda_,
                               max_iter, rel_tol);
            // compute elastic net estimates
            coef0_ = beta;
            // rescale coef back
            rescale_coef();
            // done
            return;
        }
        // else alpha > 0
        arma::mat grad_zero { arma::abs(gradient(inner)) };
        // large enough lambda for all-zero coef (except intercept terms)
        // excluding variable with zero penalty factor
        grad_zero = grad_zero.tail_rows(p0_);
        l1_lambda_max_ = grad_zero.max();
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::umat is_active_strong { arma::zeros<arma::umat>(p1_, km1_) };
        if (intercept_) {
            // only need to estimate intercept
            is_active_strong.row(0) = arma::ones<arma::umat>(1, km1_);
            run_cmd_active_cycle(beta, inner, is_active_strong,
                                 l1_lambda_max_, l2_lambda_,
                                 false, max_iter, rel_tol, verbose);
        }
        // early exit for lambda greater than lambda_max
        if (l1_lambda_ >= l1_lambda_max_) {
            coef0_ = beta;
            rescale_coef();
            return;
        }
        // use the input start if correctly specified
        if (start.size() == x_.size()) {
            if (standardize_) {
                beta = rev_rescale_coef(start);
            } else {
                beta = start;
            }
        }
        arma::umat is_active_strong_new { is_active_strong };
        bool varying_active_set { false };
        // update active set by strong rule
        arma::mat grad_beta { arma::abs(gradient(inner)) };
        double strong_rhs { 2 * l1_lambda_ - l1_lambda_max_ };
        for (size_t j { 0 }; j < km1_; ++j) {
            for (size_t l { int_intercept_ }; l < p1_; ++l) {
                if (grad_beta(l, j) >= strong_rhs) {
                    is_active_strong(l, j) = 1;
                } else {
                    beta(l, j) = 0;
                }
            }
        }
        bool kkt_failed { true };
        strong_rhs = l1_lambda_;
        // eventually, strong rule will guess correctly
        while (kkt_failed) {
            // update beta
            run_cmd_active_cycle(beta, inner, is_active_strong,
                                 l1_lambda_, l2_lambda_,
                                 varying_active_set,
                                 max_iter, rel_tol, verbose);
            // check kkt condition
            for (size_t j { 0 }; j < km1_; ++j) {
                for (size_t l { int_intercept_ }; l < p1_; ++l) {
                    if (is_active_strong(l, j) > 0) {
                        continue;
                    }
                    if (std::abs(cmd_gradient(inner, l, j)) > strong_rhs) {
                        // update active set
                        is_active_strong_new(l, j) = 1;
                    }
                }
            }
            if (l1_norm(is_active_strong - is_active_strong_new) > 0) {
                is_active_strong = is_active_strong_new;
            } else {
                kkt_failed = false;
            }
        }
        // compute elastic net estimates
        coef0_ = beta;
        // rescale coef back
        rescale_coef();
    }

    // for a sequence of lambda's
    // lambda * (alpha * lasso + (1 - alpha) / 2 * ridge)
    inline void LogisticNet::path(
        const arma::vec& lambda,
        const double alpha,
        const unsigned int nlambda,
        const double lambda_min_ratio,
        const unsigned int max_iter,
        const double rel_tol,
        const double pmin,
        const bool verbose
        )
    {
        // check alpha
        if ((alpha < 0) || (alpha > 1)) {
            throw std::range_error("The 'alpha' must be between 0 and 1.");
        }
        alpha_ = alpha;
        if ((pmin < 0) || (pmin > 1)) {
            throw std::range_error("The 'pmin' must be between 0 and 1.");
        }
        pmin_ = pmin;
        // initialize
        arma::vec one_inner { arma::zeros(n_obs_) };
        arma::mat one_beta { arma::zeros(p1_, km1_) };
        arma::mat one_grad_beta;
        double lambda_max { 0.0 };
        const bool is_ridge_only { isAlmostEqual(alpha, 0.0) };
        // if alpha = 0 and lambda is specified
        if (is_ridge_only && ! lambda.empty()) {
            l1_lambda_max_ = 0.0;
            lambda_path_ = arma::reverse(arma::unique(lambda));
        } else {
            one_grad_beta = arma::abs(gradient(one_inner));
            // large enough lambda for all-zero coef (except intercept terms)
            lambda_max = one_grad_beta.max() / std::max(alpha, 1e-2);
            // set up lambda sequence
            if (lambda.empty()) {
                double log_lambda_max { std::log(lambda_max) };
                lambda_path_ = arma::exp(
                    arma::linspace(log_lambda_max,
                                   log_lambda_max + std::log(lambda_min_ratio),
                                   nlambda)
                    );
            } else {
                lambda_path_ = arma::reverse(arma::unique(lambda));
            }
        }
        // initialize the estimate cube
        coef_path_ = arma::cube(p1_, km1_, lambda_path_.n_elem);
        // for ridge penalty
        if (is_ridge_only) {
            for (size_t li { 0 }; li < lambda_path_.n_elem; ++li) {
                run_cmd_full_cycle(one_beta, one_inner,
                                   0.0, lambda_path_(li),
                                   max_iter, rel_tol);
                coef0_ = one_beta;
                coef_path_.slice(li) = rescale_coef(coef0_);
            }
        } else {
            // if l1_lambda > 0
            double one_strong_rhs { 0.0 };
            l1_lambda_max_ = lambda_max * alpha;
            l2_lambda_ = 0.5 * lambda_max * (1 - alpha);
            // get the solution (intercepts) of l1_lambda_max for a warm start
            arma::umat is_active_strong { arma::zeros<arma::umat>(p1_, km1_) };
            if (intercept_) {
                // only need to estimate intercept
                is_active_strong.row(0) = arma::ones<arma::umat>(1, km1_);
                run_cmd_active_cycle(one_beta, one_inner, is_active_strong,
                                     l1_lambda_max_, l2_lambda_,
                                     false, max_iter, rel_tol, verbose);
            }
            // optim with varying active set when p > n
            bool varying_active_set { true };
            double old_l1_lambda { l1_lambda_max_ }; // for strong rule
            // main loop: for each lambda
            for (size_t li { 0 }; li < lambda_path_.n_elem; ++li) {
                lambda_ = lambda_path_(li);
                l1_lambda_ = lambda_ * alpha;
                l2_lambda_ = 0.5 * lambda_ * (1 - alpha);
                // early exit for lambda greater than lambda_max
                if (l1_lambda_ >= l1_lambda_max_ && alpha > 0) {
                    coef_path_.slice(li) = rescale_coef(one_beta);
                    continue;
                }
                // update active set by strong rule
                one_grad_beta = arma::abs(gradient(one_inner));
                one_strong_rhs = 2 * l1_lambda_ - old_l1_lambda;
                old_l1_lambda = l1_lambda_;
                for (size_t j { 0 }; j < km1_; ++j) {
                    for (size_t l { int_intercept_ }; l < p1_; ++l) {
                        if (one_grad_beta(l, j) >= one_strong_rhs) {
                            is_active_strong(l, j) = 1;
                        } else {
                            one_beta(l, j) = 0;
                        }
                    }
                }
                arma::umat is_active_strong_new { is_active_strong };
                bool kkt_failed { true };
                one_strong_rhs = l1_lambda_;
                // eventually, strong rule will guess correctly
                while (kkt_failed) {
                    // update beta
                    run_cmd_active_cycle(one_beta, one_inner, is_active_strong,
                                         l1_lambda_, l2_lambda_,
                                         varying_active_set,
                                         max_iter, rel_tol, verbose);
                    // check kkt condition
                    for (size_t j { 0 }; j < km1_; ++j) {
                        for (size_t l { int_intercept_ }; l < p1_; ++l) {
                            if (is_active_strong(l, j) > 0) {
                                continue;
                            }
                            if (std::abs(cmd_gradient(one_inner, l, j)) >
                                one_strong_rhs) {
                                // update active set
                                is_active_strong_new(l, j) = 1;
                            }
                        }
                    }
                    if (l1_norm(is_active_strong - is_active_strong_new) > 0) {
                        if (verbose) {
                            Rcpp::Rcout << "\nThe strong rule failed."
                                        << "\nOld active set:\n";
                            Rcpp::Rcout << is_active_strong << std::endl;
                            Rcpp::Rcout << "\nNew active set:\n";
                            Rcpp::Rcout << is_active_strong_new << std::endl;
                        }
                        is_active_strong = is_active_strong_new;
                    } else {
                        if (verbose) {
                            Rcpp::Rcout << "\nThe strong rule worked.\n";
                        }
                        kkt_failed = false;
                    }
                }
                coef0_ = one_beta;
                coef_path_.slice(li) = rescale_coef(coef0_);
            }
        }
    }


}

#endif
