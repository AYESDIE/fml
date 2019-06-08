//
// Created by ayesdie on 7/6/19.
//

#ifndef FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#define FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP

#include "../../core.hpp"

namespace fml {
namespace regression {

/**
 * A class for Linear Regression.
 */
class LinearRegression {
public:
  /**
   * Constructor for Linear Regression.
   *
   * @param dataset - Dataset of features.
   * @param labels - Set of labels corresponding to features.
   */
  LinearRegression(const xt::xarray<double>& dataset,
                   const xt::xarray<double>& labels);

  /**
   * Computes labels for given `dataset`.
   * @param dataset - Dataset of features.
   * @param labels - Evaluated set of labels corresponding to features.
   */
  void Compute(const xt::xarray<double>& dataset,
               xt::xarray<double>& labels);

private:
  /// Parameters
  xt::xarray<double> parameters;
};

} // namespace regression
} // namespace fml

#endif //FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
