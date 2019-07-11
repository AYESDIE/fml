//
// Created by ayesdie on 4/7/19.
//

#ifndef FML_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_HPP
#define FML_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_HPP

#include "fml/core.hpp"

namespace fml {
namespace regression {

template <typename DatasetType = xt::xtensor<double, 2>,
          typename LabelsType = xt::xtensor<size_t, 2>>
class SoftmaxRegressionFunction
{
public:
  SoftmaxRegressionFunction(DatasetType& dataset,
                            LabelsType& labels,
                            size_t numClasses,
                            double lambda = 0);

  /**
   * This evaluates the loss for given set of `parameters`.
   *
   * @param parameters - Parameters for Softmax Regression Function.
   * @return - Loss.
   */
  template <typename E>
  double Evaluate(const E& parameters);

  /**
   * This evaluates the gradient for the given set of `parameters`.
   *
   * @param parameters - Parameters for Softmax Regression Function.
   * @param gradient - Evaluated gradient.
   */
  // template <typename E, typename G>
  // void Gradient(const E& parameters,
  //               G& gradient);
  
  /**
   * Number of functions in the given dataset.
   */
  size_t numFunctions();

  /**
   * Returns initial points for the parameters.
   */
  xt::xtensor<double, 2> GetInitialPoints();

private:
  /// Dataset
  DatasetType dataset;

  /// Labels
  LabelsType labels;

  /// Ground Truth Matrix
  xt::xtensor<size_t, 2> groundTruth;

  /// Number of classes
  size_t numClasses;

  /// L2 Regularization constant
  double lambda;
};

} // namespace regression
} // namespace fml


#include "softmax_regression_function_impl.hpp"
#endif //FML_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_HPP