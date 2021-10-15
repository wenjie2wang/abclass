#ifndef ABCLASS_UTILS_H
#define ABCLASS_UTILS_H

#include <vector>
#include <RcppArmadillo.h>
#include "simplex.h"
#include "string.h"

namespace Abclass {

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
    // function computing relateive tolerance based on l1_norm
    template <typename T>
    inline double rel_l1_norm(const T& x_old, const T& x_new)
    {
        double denom { l1_norm(x_new + x_old) };
        if (isAlmostEqual(denom, 0)) {
            return 0;
        }
        return l1_norm(x_new - x_old) / denom;
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

    template <typename T>
    inline void msg(const T& m)
    {
        Rcpp::Rcout << m << std::endl;
    }

}


#endif
