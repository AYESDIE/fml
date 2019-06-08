//
// Created by ayesdie on 7/6/19.
//

#ifndef FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP
#define FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP

#include "../../core.hpp"

namespace fml {
namespace regression {

/**
 * This is the implementation of the Linear Regression Function.
 */
class LinearRegressionFunction
{
public:
  /**
   * Constructor for LinearRegressionFunction
   *
   * @param dataset - dataset of features.
   * @param labels - set of labels corresponding to features.
   */
  LinearRegressionFunction(const xt::xarray<double>& dataset,
                           const xt::xarray<double>& labels);

  /**
   * This evaluates the loss for given set of `parameters` using
   * squared error function.
   *
   * @param parameters - Parameters for Linear Regression Function.
   * @return - Loss.
   */
  double Evaluate(const xt::xarray<double>& parameters);

  /**
   * This evaluates the gradient for the given set of `parameters`.
   *
   * @param parameters - Parameters for Linear Regression Function.
   * @param gradient - Evaluated gradient.
   */
  void Gradient(const xt::xarray<double>& parameters,
                xt::xarray<double>& gradient);

  /**
   * Number of functions in the given dataset.
   */
  size_t numFunctions();

  /**
   * Returns initial points for the parameters.
   */
  xt::xarray<double> GetInitialPoints();

private:
  /// Dataset
  xt::xarray<double> dataset;

  /// Labels
  xt::xarray<double> labels;
};

} // namespace regression
} // namespace fml

#endif //FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP
