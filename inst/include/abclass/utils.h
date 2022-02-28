//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2022 Eli Lilly and Company
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

    // function template for crossprod of two matrix-like objects
    template <typename T_matrix_like>
    inline arma::mat crossprod(const T_matrix_like& X,
                               const T_matrix_like& Y)
    {
        return X.t() * Y;
    }
    template <typename T_matrix_like>
    inline arma::mat crossprod(const T_matrix_like& X)
    {
        return X.t() * X;
    }
    inline double innerprod(const arma::vec& x, const arma::vec& y)
    {
        return arma::as_scalar(crossprod(x, y));
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

    // soft-thresholding operator
    inline double soft_threshold(const double beta, const double lambda)
    {
        double tmp { std::abs(beta) - lambda };
        if (tmp < 0) {
            return 0;
        }
        return tmp * sign(beta);
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

    // function check convergence
    template <typename T>
    inline double rel_diff(const T& x_old, const T& x_new)
    {
        T tmp_mat { arma::abs(x_new - x_old) / (1.0 + arma::abs(x_old)) };
        return tmp_mat.max();
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

    // capped exponential
    inline arma::vec cap_exp(arma::vec x,
                             const double positive_cap = 10,
                             const double negative_cap = - 10)
    {
        x.elem(arma::find(x > positive_cap)).fill(positive_cap);
        x.elem(arma::find(x < negative_cap)).fill(negative_cap);
        return arma::exp(x);
    }
    inline double cap_exp(double x,
                          const double positive_cap = 10,
                          const double negative_cap = - 10)
    {
        return std::exp(std::min(std::max(x, negative_cap), positive_cap));
    }

    template<typename T>
    T* ptr(T& obj) { return &obj; } //turn reference into pointer

    template<typename T>
    T* ptr(T* obj) { return obj; } //obj is already pointer, return it


    template <typename T>
    inline void msg(const T& m)
    {
        Rcpp::Rcout << m << "\n";
    }

}


#endif
