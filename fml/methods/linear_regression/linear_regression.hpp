//
// Created by ayesdie on 7/6/19.
//

#ifndef FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#define FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP

#include "fml/core.hpp"
#include "linear_regression_function.hpp"
#include "fml/core/optimizers/gradient_descent/gradient_descent.hpp"

namespace fml {
namespace regression {

/**
 * A class for Linear Regression.
 */
template <typename DatasetType = xt::xarray<double>,
          typename LabelsType = xt::xarray<double>>
class LinearRegression {
public:
  /**
   * Constructor for Linear Regression.
   *
   * @param dataset - Dataset of features.
   * @param labels - Set of labels corresponding to features.
   */
  template <typename OptimizerType>
  LinearRegression(DatasetType& dataset,
                   LabelsType& labels,
                   OptimizerType& optimizer = fml::optimizer::GradientDescent());

  /**
   * Computes labels for given `dataset`.
   * @param dataset - Dataset of features.
   * @param labels - Evaluated set of labels corresponding to features.
   */
  void Compute(DatasetType& dataset,
               LabelsType& labels);

private:
  /// Parameters
  xt::xtensor<double, 2> parameters;
};

} // namespace regression
} // namespace fml

#include "linear_regression_impl.hpp"

#endif //FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
