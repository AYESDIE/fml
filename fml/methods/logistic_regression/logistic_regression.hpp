//
// Created by ayesdie on 10/6/19.
//

#ifndef FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
#define FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP

#include "fml/core/optimizers/gradient_descent/gradient_descent.hpp"
#include "logistic_regression_function.hpp"
#include "fml/core.hpp"

namespace fml {
namespace regression {

/**
 * A class for Logistic Regression.
 */
template <typename DatasetType = xt::xtensor<double, 2>,
          typename LabelsType = xt::xtensor<size_t, 2>>
class LogisticRegression{
public:
  /**
   * Constructor for Logistic Regression.
   *
   * @param dataset - Dataset of features.
   * @param labels - Set of labels corresponding to features.
   * @param optimizer - Optimizer used for optimization.
   */
  template <typename OptimizerType>
  LogisticRegression(const DatasetType& dataset,
                     const LabelsType& labels,
                     OptimizerType& optimizer = fml::optimizer::GradientDescent());

  /**
   * Computes labels for given `dataset`.
   * @param dataset - Dataset of features.
   * @param labels - Evaluated set of labels corresponding to features.
   */
  void Compute(const DatasetType& dataset,
               LabelsType& labels);

private:
  /// Parameters.
  xt::xtensor<double, 2> parameters;
};

} // namespace regression
} // namespace fml

#include "logistic_regression_impl.hpp"

#endif //FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
