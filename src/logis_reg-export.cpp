#include <cmath>
#include <RcppArmadillo.h>
#include <malc.h>


// [[Rcpp::export]]
Rcpp::List rcpp_logistic_reg(
    const arma::mat& x,
    const arma::uvec& y,
    const double lambda,
    const double alpha,
    const arma::mat& start,
    const arma::vec& weight,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int max_iter = 200,
    const double rel_tol = 1e-3,
    const double pmin = 1e-5,
    const bool verbose = false
    )
{
    Malc::LogisticReg object {
        x, y, intercept, standardize, weight
    };
    object.elastic_net(lambda, alpha,
                       start, max_iter, rel_tol, pmin, verbose);
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("weight") = Malc::arma2rvec(object.get_weight()),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") = lambda,
            Rcpp::Named("alpha") = alpha,
            Rcpp::Named("l1_lambda") = object.l1_lambda_,
            Rcpp::Named("l2_lambda") = object.l2_lambda_,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_
            )
        );
}


// [[Rcpp::export]]
Rcpp::List rcpp_logistic_path(
    const arma::mat& x,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const arma::vec& weight,
    const unsigned int nfolds = 0,
    const bool stratified = true,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int max_iter = 200,
    const double rel_tol = 1e-3,
    const double pmin = 1e-5,
    const bool verbose = false
    )
{
    Malc::LogisticReg object {
        x, y, intercept, standardize, weight
    };
    object.elastic_net_path(lambda, alpha, nlambda, lambda_min_ratio,
                            nfolds, stratified,
                            max_iter, rel_tol, pmin, verbose);
    Rcpp::NumericVector lambda_vec { Malc::arma2rvec(object.lambda_path_) };
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_path_,
        Rcpp::Named("weight") = Malc::arma2rvec(object.get_weight()),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") = lambda_vec,
            Rcpp::Named("alpha") = alpha,
            Rcpp::Named("l1_lambda") = alpha * lambda_vec,
            Rcpp::Named("l2_lambda") = 0.5 * (1 - alpha) * lambda_vec,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_
            ),
        Rcpp::Named("cross_validation") = Rcpp::List::create(
            Rcpp::Named("miss_number") = object.cv_miss_number_,
            Rcpp::Named("accuracy") = object.cv_accuracy_
            )
        );
}

// [[Rcpp::export]]
Rcpp::List rcpp_logistic_et(
    const arma::mat& x,
    const arma::uvec& y,
    const arma::vec& lambda,
    const double alpha,
    const unsigned int nlambda,
    const double lambda_min_ratio,
    const arma::vec& weight,
    const bool intercept = true,
    const bool standardize = true,
    const unsigned int max_iter = 50,
    const double rel_tol = 1e-3,
    const double pmin = 1e-5,
    const bool verbose = false
    )
{
    Malc::LogisticReg object {
        x, y, intercept, standardize, weight
    };
    object.et_tune_net(lambda, alpha, nlambda, lambda_min_ratio,
                       max_iter, rel_tol, pmin, verbose);
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = object.coef_,
        Rcpp::Named("weight") = Malc::arma2rvec(object.get_weight()),
        Rcpp::Named("regularization") = Rcpp::List::create(
            Rcpp::Named("lambda") = Malc::arma2rvec(object.lambda_path_),
            Rcpp::Named("alpha") = alpha,
            Rcpp::Named("l1_lambda_max") = object.l1_lambda_max_
            ),
        Rcpp::Named("tuned") = Rcpp::List::create(
            Rcpp::Named("lambda") = object.lambda_,
            Rcpp::Named("l1_lambda") = object.l1_lambda_,
            Rcpp::Named("l2_lambda") = object.l2_lambda_
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
    Malc::Simplex sim { k };
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
        Rcpp::Named("predicted", Malc::arma2rvec(max_idx)),
        Rcpp::Named("accuracy", acc)
        );
}
