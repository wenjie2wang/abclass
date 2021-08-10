#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>
#include "simplex.h"

namespace Malc {

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
        } else if (x > 0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    // positive part
    template <typename T_scalar>
    inline T_scalar positive(T_scalar x)
    {
        if (x < 0) {
            return 0;
        } else {
            return x;
        }
    }
    // soft-thresholding operator
    inline double soft_threshold(const double beta, const double lambda)
    {
        return positive(std::abs(beta) - lambda) * sign(beta);
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
        } else {
            return l1_norm(x_new - x_old) / denom;
        }
    }


}  // Malc


#endif /* UTILS_H */
