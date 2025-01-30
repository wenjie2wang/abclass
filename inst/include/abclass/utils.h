//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2025 Eli Lilly and Company
//
// This file is part of the R package abclass.
//
// The R package abclass is free software: You can redistribute it and/or
// modify it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or any later
// version (at your option). See the GNU General Public License at
// <https://www.gnu.org/licenses/> for details.
//
// The R package abclass is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//

#ifndef ABCLASS_UTILS_H
#define ABCLASS_UTILS_H

#include <cmath>
#include <vector>
#include <RcppArmadillo.h>

namespace abclass {

    // convert arma vector type to Rcpp vector type
    template <typename T>
    inline Rcpp::NumericVector arma2rvec(const T& x) {
        return Rcpp::NumericVector(x.begin(), x.end());
    }
    // convert Rcpp::NumericVector to arma::colvec
    template <typename T>
    inline arma::vec rvec2arma(const T& x) {
        return arma::vec(x.begin(), x.size(), false);
    }
    inline arma::vec mat2vec(const arma::mat& x)
    {
        return arma::conv_to<arma::vec>::from(x);
    }

    // positive part
    inline double positive_part(const double x)
    {
        if (x > 0) {
            return x;
        }
        return 0.0;
    }
    inline arma::vec positive_part(const arma::vec& x)
    {
        arma::vec out = x;
        arma::vec::iterator it = out.begin();
        arma::vec::iterator it_end = out.end();
        for (; it != it_end; ++it) {
            *it = positive_part(*it);
        }
        return out;
    }

    // compare double-precision numbers for almost equality
    inline bool isAlmostEqual(double A, double B)
    {
        double MaxRelDiff {std::numeric_limits<double>::epsilon()};
        // compute the difference.
        double diff = std::abs(A - B);
        A = std::abs(A);
        B = std::abs(B);
        // Find the largest
        double largest = (B > A) ? B : A;
        if (diff <= largest * MaxRelDiff) {
            return true;
        } else {
            return false;
        }
    }
    inline bool is_gt(double A, double B)
    {
        return (! isAlmostEqual(A, B)) && (A > B);
    }
    inline bool is_lt(double A, double B)
    {
        return (! isAlmostEqual(A, B)) && (A < B);
    }
    inline bool is_ge(double A, double B)
    {
        return ! is_lt(A, B);
    }
    inline bool is_le(double A, double B)
    {
        return ! is_gt(A, B);
    }

    // function that computes L1-norm
    template <typename T>
    inline double l1_norm(const T& x)
    {
        return arma::accu(arma::abs(x));
    }
    template <typename T>
    inline double l2_norm_square(const T& x)
    {
        return arma::accu(arma::square(x));
    }
    template <typename T>
    inline double l2_norm(const T& x)
    {
        return std::sqrt(l2_norm_square(x));
    }

    // sign function
    inline double sign(const double x)
    {
        if (x < 0) {
            return - 1.0;
        }
        if (x > 0) {
            return 1.0;
        }
        return 0.0;
    }
    inline arma::vec sign(const arma::vec& x)
    {
        arma::vec res { arma::zeros(x.n_elem) };
        for (size_t i {0}; i < x.n_elem; ++i) {
            res[i] = sign(x[i]);
        }
        return res;
    }

    // soft-thresholding operator
    inline double soft_threshold(const double beta, const double lambda)
    {
        double tmp { std::abs(beta) - lambda };
        if (tmp < 0) {
            return 0;
        }
        return tmp * sign(beta);
    }
    // inline arma::rowvec soft_threshold(const arma::rowvec& beta,
    //                                    const double l2_beta,
    //                                    const double lambda)
    // {
    //     double tmp { 1 - lambda  / l2_beta };
    //     if (tmp <= 0) {
    //         return arma::zeros<arma::rowvec>(beta.n_elem);
    //     }
    //     return tmp * beta;
    // }

    // function check convergence
    template <typename T>
    inline double rel_diff(const T& x_old, const T& x_new)
    {
        T tmp { arma::abs(x_new - x_old) / (1.0 + arma::abs(x_new)) };
        return tmp.max();
    }
    template <typename T>
    inline double l1_sum_diff(const T& x_old, const T& x_new)
    {
        T tmp { arma::abs(x_new - x_old) };
        return arma::accu(tmp);
    }

    // set difference for vector a and vector b
    template <typename T, template <typename> class ARMA_VEC_TYPE>
    inline ARMA_VEC_TYPE<T> vec_diff(const ARMA_VEC_TYPE<T>& a,
                                     const ARMA_VEC_TYPE<T>& b)
    {
        std::vector<T> res;
        ARMA_VEC_TYPE<T> s_a { arma::sort(a) };
        ARMA_VEC_TYPE<T> s_b { arma::sort(b) };
        std::set_difference(s_a.begin(), s_a.end(),
                            s_b.begin(), s_b.end(),
                            std::inserter(res, res.begin()));
        return arma::sort(arma::conv_to<ARMA_VEC_TYPE<T>>::from(res));
    }

    inline void msg() {
        Rcpp::Rcout << "\n";
    }
    template <typename T1, typename ... T2>
    inline void msg(const T1& m1, const T2&... m2)
    {
        Rcpp::Rcout << m1;
        msg(m2...);
    }

    // compute a * b for sake of numerical stability
    inline double exp_log_sum(const double a, const double b)
    {
        return std::exp(std::log(a) + std::log(b));
    }

    // FIXME select rows: remedy for sparse matrices
    inline arma::mat subset_rows(const arma::mat& mat,
                                 const arma::uvec& row_index)
    {
        return mat.rows(row_index);
    }
    inline arma::sp_mat subset_rows(const arma::sp_mat& mat,
                                    const arma::uvec& row_index)
    {
        arma::sp_mat out { mat.t() };
        return out.cols(row_index).t();
    }

    // column-wise standard deviations
    template <typename T>
    inline arma::rowvec col_sd(const T& mat)
    {
        arma::rowvec out { arma::var(mat, 1) };
        return arma::sqrt(out);
    }

    // MCP penalty function for theta >= 0
    inline double mcp_penalty(const double theta,
                              const double lambda,
                              const double gamma)
    {
        if (theta < gamma * lambda) {
            return theta * (lambda - 0.5 * theta / gamma);
        }
        return 0.5 * gamma * lambda * lambda;
    }
    // first derivative of MCP (wrt theta) for theta >= 0
    inline double dmcp_penalty(const double theta,
                               const double lambda,
                               const double gamma)
    {
        // const double numer { gamam * lambda - theta};
        const double numer { lambda - theta / gamma };
        if (numer > 0) {
            // return numer / gamma;
            return numer;
        }
        return 0.0;
    }

    // exponential penalty function (for theta >= 0)
    inline double exp_penalty(const double theta,
                              const double lambda,
                              const double tau)
    {
        return std::pow(lambda, 2) / tau *
            (1 - std::exp(- tau * theta / lambda));
    }
    // first derivative of exponential penalty (wrt theta)
    // assume lambda > 0
    inline double dexp_penalty(const double theta,
                               const double lambda,
                               const double tau)
    {
        return lambda * std::exp(- theta * tau / lambda);
    }

}


#endif
