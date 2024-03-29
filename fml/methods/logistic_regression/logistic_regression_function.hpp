//
// Created by ayesdie on 10/6/19.
//

#ifndef FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
#define FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP

#include "fml/core.hpp"

namespace fml {
namespace regression {

/**
 * This is the implementation of L2 Regularized Logistic Regression
 * Function.
 */
template <typename DatasetType = xt::xtensor<double, 2>,
          typename LabelsType = xt::xtensor<size_t, 2>>
class LogisticRegressionFunction
{
public:
  /**
   * Constructor for LogisticRegressionFunction.
   *
   * @param dataset - dataset of features.
   * @param labels - set of labels corresponding to features.
   * @param lambda - L2 regularization parameter.
   */
  LogisticRegressionFunction(DatasetType& dataset,
                             LabelsType& labels,
                             const double lambda = 0);

  /**
   * This evaluates the loss for given set of `parameters`.
   *
   * @param parameters - Parameters for Linear Regression Function.
   * @return - Loss.
   */
  template <typename E>
  double Evaluate(const E& parameters);

  /**
   * This evaluates the loss for given batch of `parameters`.
   *
   * @param parameters - Parameters for Linear Regression Function.
   * @param firstId - Index for first element.
   * @param batchSize - Size of batch.
   * @return - Loss.
   */
  template <typename E>
  double Evaluate(const E& parameters,
                  const size_t& firstId,
                  const size_t& batchSize);

  /**
   * This evaluates the gradient for the given set of `parameters`.
   *
   * @param parameters - Parameters for Linear Regression Function.
   * @param gradient - Evaluated gradient.
   */
  template <typename E, typename G>
  void Gradient(const E& parameters,
                G& gradient);

  /**
   * This evaluates the gradient for the given batch of `parameters`.
   * @param parameters - Parameters for Linear Regression Function.
   * @param firstId - Index for first element.
   * @param batchSize - Size of batch.   
   * @param gradient - Evaluated gradient.
   */
  template <typename E, typename G>
  void Gradient(const E& parameters,
                const size_t& firstId,
                G& gradient,
                const size_t& batchSize);

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

  /// L2 Regularization constant
  double lambda;
};

} // namespace regression
} // namespace fml

#include "logistic_regression_function_impl.hpp"

#endif //FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
