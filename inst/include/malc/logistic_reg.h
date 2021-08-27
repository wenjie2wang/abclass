#ifndef MALC_LOGISTIC_REG_H
#define MALC_LOGISTIC_REG_H

#include <vector>
#include <RcppArmadillo.h>
#include "utils.h"
#include "cross-validation.h"

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
        arma::vec obs_weight_;  // optional observation weights: of length n
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
        // for one lambda
        arma::mat coef0_;         // coef (not scaled for the origin x_)
        // arma::mat en_coef0_;      // (not scaled) elastic net estimates

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
        // class conditional probability matrix: n by k
        arma::mat coef_;
        // arma::mat prob_mat_;    // for coef_
        // arma::mat en_coef_;
        // arma::mat en_prob_mat_; // for en_coef_
        // for a lambda sequence
        arma::vec lambda_path_;   // lambda sequence
        arma::cube coef_path_;
        // arma::cube prob_path_;
        arma::mat cv_miss_number_;
        arma::mat cv_accuracy_;
        // arma::cube en_coef_path_;
        // arma::cube en_prob_path_;
        arma::mat cv_en_miss_number_;
        arma::mat cv_en_accuracy_;

        // default constructor
        LogisticReg() {}

        //! @param x_ The design matrix without an intercept term.
        //! @param y The category index vector.
        LogisticReg(const arma::mat& x,
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
            k_ = arma::max(y_);
            km1_ = k_ - 1;
            n_obs_ = x_.n_rows;
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
            // set the CMD lowerbound (which needs to be done only once)
            set_cmd_lowerbound();
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
            // en_coef_ = rescale_coef(en_coef0_);
        }
        // compute cov lowerbound used in regularied model
        inline void set_cmd_lowerbound()
        {
            arma::mat sqx { arma::square(x_) };
            sqx.each_col() %= obs_weight_;
            cmd_lowerbound_ = arma::sum(sqx, 0) / (4 * n_obs_);
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
                out += obs_weight_(i) * vertex_mat_(i, j) * x_(i, l) * p_est;
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
        // class conditional probability
        inline arma::mat compute_prob_mat(const arma::mat& beta,
                                          const arma::mat& x) const
        {
            arma::mat out { x * beta };
            out *= vertex_.t();
            for (size_t j { 0 }; j < k_; ++j) {
                out.col(j)  = 1 / loss_derivative(out.col(j));
            }
            arma::vec row_sums { arma::sum(out, 1) };
            for (size_t j { 0 }; j < k_; ++j) {
                out.col(j) /= row_sums;
            }
            return out;
        }
        inline arma::mat compute_prob_mat(const arma::mat& beta) const
        {
            return compute_prob_mat(beta, x_);
        }
        // inline void set_prob_mat()
        // {
        //     prob_mat_ = compute_prob_mat(coef0_);
        // }
        // inline void set_en_prob_mat()
        // {
        //     en_prob_mat_ = compute_prob_mat(en_coef0_);
        // }

        // predict categories for the training set
        inline arma::uvec predict_cat(const arma::mat& prob_mat) const
        {
            return arma::index_max(prob_mat, 1) + 1;
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
        // inline double accuracy() const
        // {
        //     arma::uvec max_idx { predict_cat(prob_mat_) };
        //     return static_cast<double>(arma::sum(max_idx == y_)) / y_.n_elem;
        // }
        // inline double en_accuracy() const
        // {
        //     arma::uvec max_idx { predict_cat(en_prob_mat_) };
        //     return static_cast<double>(arma::sum(max_idx == y_)) / y_.n_elem;
        // }

        // run one cycle of coordinate descent over a given active set
        inline void run_one_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::umat& is_active,
                                         const double l1_lambda,
                                         const double l2_lambda,
                                         const bool update_active,
                                         const bool early_stop,
                                         const bool verbose);
        // run a complete cycle of CMD for a given active set and given lambda's
        inline void run_cmd_active_cycle(arma::mat& beta,
                                         arma::vec& inner,
                                         arma::umat& is_active,
                                         const double l1_lambda,
                                         const double l2_lambda,
                                         const bool varying_active_set,
                                         const unsigned int max_iter,
                                         const double rel_tol,
                                         const bool early_stop,
                                         const bool verbose);

        // for a perticular lambda
        inline void elastic_net(const double lambda,
                                const double alpha,
                                const arma::mat& start,
                                const unsigned int max_iter,
                                const double rel_tol,
                                const double pmin,
                                const bool early_stop,
                                const bool verbose);

        // for a sequence of lambda's
        inline void elastic_net_path(const arma::vec& lambda,
                                     const double alpha,
                                     const unsigned int nlambda,
                                     const double lambda_min_ratio,
                                     const unsigned int nfolds,
                                     const bool stratified,
                                     const unsigned int max_iter,
                                     const double rel_tol,
                                     const double pmin,
                                     const bool early_stop,
                                     const bool verbose);

        // getters
        inline arma::vec get_weight() const
        {
            return obs_weight_;
        }

    };                          // end of class

    inline void LogisticReg::run_one_active_cycle(
        arma::mat& beta,
        arma::vec& inner,
        arma::umat& is_active,
        const double l1_lambda,
        const double l2_lambda,
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
                        cmd_lowerbound_(l) * beta(l, j) - dlj, l1_lambda
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
                                         l1_lambda, l2_lambda, true,
                                         early_stop, verbose);
                    if (rel_l1_norm(beta, beta0) < rel_tol) {
                        break;
                    }
                    beta0 = beta;
                    ii++;
                }
                // run a full cycle over the converged beta
                run_one_active_cycle(beta, inner, is_active_new,
                                     l1_lambda, l2_lambda, true,
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
                                     l1_lambda, l2_lambda, false,
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
    // lambda_1 * lasso + lambda_2 * ridge
    inline void LogisticReg::elastic_net(
        const double lambda,
        const double alpha,
        const arma::mat& start,
        const unsigned int max_iter,
        const double rel_tol,
        const double pmin,
        const bool early_stop,
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
        arma::mat grad_zero { arma::abs(gradient(inner)) };
        arma::mat grad_beta { grad_zero }, strong_rhs { grad_beta };
        // large enough lambda for all-zero coef (except intercept terms)
        // excluding variable with zero penalty factor
        grad_zero = grad_zero.tail_rows(p0_);
        l1_lambda_max_ = arma::max(arma::conv_to<arma::vec>::from(grad_zero));
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::umat is_active_strong { arma::zeros<arma::umat>(p1_, km1_) };
        if (intercept_) {
            // only need to estimate intercept
            is_active_strong.row(0) = arma::ones<arma::umat>(1, km1_);
            run_cmd_active_cycle(beta, inner, is_active_strong,
                                 l1_lambda_max_, l2_lambda_,
                                 false, max_iter, rel_tol, early_stop, verbose);
        }
        // early exit for lambda greater than lambda_max
        if (l1_lambda_ >= l1_lambda_max_) {
            coef0_ = beta;
            // en_coef0_ = beta;
            // set_prob_mat();
            // en_prob_mat_ = prob_mat_;
            rescale_coef();
            return;
        }

        // use the input start if correctly specified
        if (start.size() == x_.size()) {
            beta = start;
        }

        // update active set by strong rule
        grad_beta = arma::abs(gradient(inner));
        strong_rhs = (2 * l1_lambda_ - l1_lambda_max_);

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
        strong_rhs = l1_lambda_;
        // eventually, strong rule will guess correctly
        while (kkt_failed) {
            // update beta
            run_cmd_active_cycle(beta, inner, is_active_strong,
                                 l1_lambda_, l2_lambda_,
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
        // compute elastic net estimates
        coef0_ = beta;
        // en_coef0_ = (1 + l2_lambda_) * coef0_;
        // en_coef0_(0) = coef0_(0);   // for intercept

        // compute probability matrix, disable now
        // set_prob_mat();
        // set_en_prob_mat();

        // rescale coef back
        rescale_coef();
    }

    // for a sequence of lambda's
    // lambda * (alpha * lasso + (1 - alpha) / 2 * ridge)
    inline void LogisticReg::elastic_net_path(
        const arma::vec& lambda,
        const double alpha,
        const unsigned int nlambda,
        const double lambda_min_ratio,
        const unsigned int nfolds,
        const bool stratified,
        const unsigned int max_iter,
        const double rel_tol,
        const double pmin,
        const bool early_stop,
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
        // for one lambda
        arma::vec one_inner { arma::zeros(n_obs_) };
        arma::mat one_beta { arma::zeros(p1_, km1_) };
        arma::mat one_grad_beta { arma::abs(gradient(one_inner)) },
            one_strong_rhs { one_grad_beta };
        // large enough lambda for all-zero coef (except intercept terms)
        double lambda_max {
            arma::max(arma::conv_to<arma::vec>::from(one_grad_beta)) /
            std::max(alpha, 1e-2)
        };
        l1_lambda_max_ = lambda_max * alpha;
        l2_lambda_ = 0.5 * lambda_max * (1 - alpha);
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
        // initialize the estimate matrix
        coef_path_ = arma::cube(p1_, km1_, lambda_path_.n_elem);
        // en_coef_path_ = coef_path_;
        // prob_path_ = arma::cube(n_obs_, k_, lambda_path_.n_elem);
        // en_prob_path_ = prob_path_;
        // get the solution (intercepts) of l1_lambda_max for a warm start
        arma::umat is_active_strong { arma::zeros<arma::umat>(p1_, km1_) };
        if (intercept_) {
            // only need to estimate intercept
            is_active_strong.row(0) = arma::ones<arma::umat>(1, km1_);
            run_cmd_active_cycle(one_beta, one_inner, is_active_strong,
                                 l1_lambda_max_, l2_lambda_,
                                 false, max_iter, rel_tol, early_stop, verbose);
        }
        // optim with varying active set when p > n
        bool varying_active_set { false };
        if (p1_ > n_obs_ || p1_ > 50) {
            varying_active_set = true;
        }
        double old_l1_lambda { l1_lambda_max_ };
        // main loop: for each lambda
        for (size_t li { 0 }; li < lambda_path_.n_elem; ++li) {
            lambda_ = lambda_path_(li);
            l1_lambda_ = lambda_ * alpha;
            l2_lambda_ = 0.5 * lambda_ * (1 - alpha);
            // early exit for lambda greater than lambda_max
            if (l1_lambda_ >= l1_lambda_max_) {
                coef_path_.slice(li) = rescale_coef(one_beta);
                // en_coef_path_.slice(li) = coef_path_.slice(li);
                // prob_path_.slice(li) = compute_prob_mat(one_beta);
                // en_prob_path_.slice(li) = prob_path_.slice(li);
                continue;
            }
            // update active set by strong rule
            one_grad_beta = arma::abs(gradient(one_inner));
            one_strong_rhs = (2 * l1_lambda_ - old_l1_lambda);
            old_l1_lambda = l1_lambda_;
            for (size_t j { 0 }; j < km1_; ++j) {
                for (size_t l { int_intercept_ }; l < p1_; ++l) {
                    if (one_grad_beta(l, j) >= one_strong_rhs(l, j)) {
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
                                     max_iter, rel_tol, early_stop, verbose);
                // check kkt condition
                for (size_t j { 0 }; j < km1_; ++j) {
                    for (size_t l { int_intercept_ }; l < p1_; ++l) {
                        if (is_active_strong(l, j)) {
                            continue;
                        }
                        if (std::abs(cmd_gradient(one_inner, l, j)) >
                            one_strong_rhs(l, j)) {
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
            // compute elastic net estimates
            coef0_ = one_beta;
            // en_coef0_ = (1 + l2_lambda_) * one_beta;
            // en_coef0_(0) = one_beta(0);
            coef_path_.slice(li) = rescale_coef(coef0_);
            // en_coef_path_.slice(li) = rescale_coef(en_coef0_);
            // compute probability matrix
            // prob_path_.slice(li) = compute_prob_mat(coef0_);
            // en_prob_path_.slice(li) = compute_prob_mat(en_coef0_);
        }
        // cross-validation
        if (nfolds > 0) {
            cv_miss_number_ = cv_en_miss_number_ =
                cv_accuracy_ = cv_en_accuracy_ =
                arma::zeros(lambda_path_.n_elem, nfolds);
            arma::uvec strata;
            if (stratified) {
                strata = y_ - 1;
            }
            CrossValidation cv_obj { n_obs_, nfolds, strata };
            for (size_t i { 0 }; i < nfolds; ++i) {
                arma::mat train_x { x_.rows(cv_obj.train_index_.at(i)) };
                if (intercept_) {
                    train_x = train_x.tail_cols(p0_);
                }
                arma::uvec train_y { y_.rows(cv_obj.train_index_.at(i)) };
                arma::mat test_x { x_.rows(cv_obj.test_index_.at(i)) };
                arma::uvec test_y { y_.rows(cv_obj.test_index_.at(i)) };
                LogisticReg reg_obj { train_x, train_y, intercept_, false };
                reg_obj.elastic_net_path(lambda_path_, alpha,
                                         nlambda, lambda_min_ratio,
                                         0, true,
                                         max_iter, rel_tol, pmin,
                                         early_stop, false);
                for (size_t l { 0 }; l < lambda_path_.n_elem; ++l) {
                    cv_miss_number_(l, i) = reg_obj.miss_number(
                        reg_obj.coef_path_.slice(l), test_x, test_y);
                    // cv_en_miss_number_(l, i) = reg_obj.miss_number(
                    //     reg_obj.en_coef_path_.slice(l), test_x, test_y);
                    cv_accuracy_(l, i) = reg_obj.accuracy(
                        reg_obj.coef_path_.slice(l), test_x, test_y);
                    // cv_en_accuracy_(l, i) = reg_obj.accuracy(
                    //     reg_obj.en_coef_path_.slice(l), test_x, test_y);
                }
            }
        }
    }

}  // Malc

#endif /* MALC_LOGISTIC_REG_H */
