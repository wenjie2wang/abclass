% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/moml.R
\name{moml}
\alias{moml}
\alias{moml.control}
\title{Multi-Category Outcome-Weighted Margin-Based Learning (MOML)}
\usage{
moml(
  x,
  treatment,
  reward,
  propensity_score,
  intercept = TRUE,
  loss = c("logistic", "boost", "hinge-boost", "lum"),
  control = moml.control(),
  ...
)

moml.control(...)
}
\arguments{
\item{x}{A numeric matrix representing the design matrix.  No missing valus
are allowed.  The coefficient estimates for constant columns will be
zero.  Thus, one should set the argument \code{intercept} to \code{TRUE}
to include an intercept term instead of adding an all-one column to
\code{x}.}

\item{treatment}{The assigned treatments represented by a character,
integer, numeric, or factor vector.}

\item{reward}{A numeric vector representing the rewards.  It is assumed that
a larger reward is more desirable.}

\item{propensity_score}{A numeric vector taking values between 0 and 1
representing the propensity score.}

\item{intercept}{A logical value indicating if an intercept should be
considered in the model.  The default value is \code{TRUE} and the
intercept is excluded from regularization.}

\item{loss}{A character value specifying the loss function.  The available
options are \code{"logistic"} for the logistic deviance loss,
\code{"boost"} for the exponential loss approximating Boosting machines,
\code{"hinge-boost"} for hybrid of SVM and AdaBoost machine, and
\code{"lum"} for largin-margin unified machines (LUM).  See Liu, et
al. (2011) for details.}

\item{control}{A list of control parameters. See \code{abclass.control()}
for details.}

\item{...}{Other arguments passed to the control function, which calls the
\code{abclass.control()} internally.}
}
\description{
Performs the outcome-weighted margin-based learning for multicategory
treatments proposed by Zhang, et al. (2020).
}
\references{
Zhang, C., Chen, J., Fu, H., He, X., Zhao, Y., & Liu, Y. (2020).
Multicategory outcome weighted margin-based learning for estimating
individualized treatment rules. Statistica Sinica, 30, 1857--1879.
}
