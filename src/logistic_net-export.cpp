#include <cmath>
#include <RcppArmadillo.h>
#include <abclass.h>


// [[Rcpp::export]]
Rcpp::List rcpp_logistic_net(
    const arma::mat& x,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const arma::vec& weight,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int max_iter = 1000,
    const double rel_tol = 1e-5,
    const bool varying_active_set = true,
    const unsigned int verbose = 0
    )
{
    abclass::LogisticNet object {
        x, y, intercept, standardize, weight
    };
    object.fit(lambda, alpha, nlambda, lambda_min_ratio,
               max_iter, rel_tol, varying_active_set, verbose);
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("weight") = abclass::arma2rvec(object.get_weight()),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") = lambda,
            Rcpp::Named("alpha") = alpha,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_
            )
        );
}

arma::vec logistic_derivative(const arma::vec& u)
{
    return - 1.0 / (1.0 + arma::exp(u));
}

// [[Rcpp::export]]
arma::mat rcpp_prob_mat(const arma::mat& beta,
                        const arma::mat& x)
{
    arma::mat out { x * beta };
    unsigned int k { beta.n_cols + 1 };
    abclass::Simplex sim { k };
    arma::mat vertex { sim.get_vertex() };
    out *= vertex.t();
    for (size_t j { 0 }; j < k; ++j) {
        out.col(j)  = 1 / logistic_derivative(out.col(j));
    }
    arma::vec row_sums { arma::sum(out, 1) };
    for (size_t j { 0 }; j < k; ++j) {
        out.col(j) /= row_sums;
    }
    return out;
}

// [[Rcpp::export]]
arma::uvec rcpp_predict_cat(const arma::mat& prob_mat)
{
    return arma::index_max(prob_mat, 1);
}

// [[Rcpp::export]]
Rcpp::List rcpp_accuracy(const arma::mat& new_x,
                         const arma::uvec& new_y,
                         const arma::mat& beta)
{
    arma::mat prob_mat { rcpp_prob_mat(beta, new_x) };
    arma::uvec max_idx { rcpp_predict_cat(prob_mat) };
    double acc;
    if (new_y.empty()) {
        acc = std::nan("1");
    } else {
        acc = static_cast<double>(arma::sum(max_idx == new_y)) / new_y.n_elem;
    }
    return Rcpp::List::create(
        Rcpp::Named("class_prob", prob_mat),
        Rcpp::Named("predicted", abclass::arma2rvec(max_idx)),
        Rcpp::Named("accuracy", acc)
        );
}
