% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/abclass_propscore.R
\name{abclass_propscore}
\alias{abclass_propscore}
\title{Estimate Propensity Score by the Angle-Based Classifiers}
\usage{
abclass_propscore(
  x,
  treatment,
  intercept = TRUE,
  weight = NULL,
  loss = c("logistic", "boost", "hinge-boost", "lum"),
  control = list(),
  tuning = c("et", "cv"),
  ...
)
}
\arguments{
\item{x}{A numeric matrix representing the design matrix.  No missing valus
are allowed.  The coefficient estimates for constant columns will be
zero.  Thus, one should set the argument \code{intercept} to \code{TRUE}
to include an intercept term instead of adding an all-one column to
\code{x}.}

\item{treatment}{The assigned treatments represented by a character,
integer, numeric, or factor vector.}

\item{intercept}{A logical value indicating if an intercept should be
considered in the model.  The default value is \code{TRUE} and the
intercept is excluded from regularization.}

\item{weight}{A numeric vector for nonnegative observation weights. Equal
observation weights are used by default.}

\item{loss}{A character value specifying the loss function.  The available
options are \code{"logistic"} for the logistic deviance loss,
\code{"boost"} for the exponential loss approximating Boosting machines,
\code{"hinge-boost"} for hybrid of SVM and AdaBoost machine, and
\code{"lum"} for largin-margin unified machines (LUM).  See Liu, et
al. (2011) for details.}

\item{control}{A list of control parameters. See \code{abclass.control()}
for details.}

\item{tuning}{A character vector specifying the tuning method.  This
argument will be ignored if a single \code{lambda} is specified through
\code{control}.}

\item{...}{Other arguments passed to the corresponding methods.}
}
\description{
A wrap function to estimate the propensity score by the multi-category
angle-based large-margin classifiers.
}
