# abclass 0.3.0

## New features

* Added experimental group-wise regularization by group SCAD and group MCP
  penalty.
* Added a new function named `abclass.control()` to specify the control
  parameters and simplify the main function interface.

## Minor changes

* Renamed the argument `max_iter` to `maxit` for `abclass()`.

## Bug fixes

* Fixed the validation indices in cross-validation procedure


# abclass 0.2.0

## New features

* Added experimental group-wise regularization by group lasso penalty.

## Minor changes

* Removed the function call from the return of `abclass()` to avoid
  unnecessarily large returned objects
* Changed the default value of `lum_c` for `abclass()` from 0 to 1.
* Renamed the argument `rel_tol` to `epsilon` for `abclass()`.

## Bug fixes

* Fixed the first derivatives of the boosting loss
* Fixed the label prediction by using the fitted inner products instead of the
  probability estimates
* Fixed the computation of regularization terms for verbose outputs in
  `AbclassNet`
* Fixed the computation of validation accuracy in cross-validation
* Fixed the assignment of `lum_c` in the associated header files.
* Fixed the computation of lower bound for distinct observation weights


# abclass 0.1.0

## New features

* The first release of **abclass** providing the multi-category angle-based
  large-margin classifiers with various loss functions.
