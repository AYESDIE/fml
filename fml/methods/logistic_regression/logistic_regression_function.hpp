//
// Created by ayesdie on 10/6/19.
//

#ifndef FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
#define FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP

#include "../../core.hpp"

namespace fml {
namespace regression {

template <typename DatasetType = xt::xtensor<double, 2>,
          typename LabelsType = xt::xtensor<size_t, 2>>
class LogisticRegressionFunction
{
public:
  LogisticRegressionFunction(DatasetType& dataset,
                             LabelsType& labels,
                             const double lambda = 0);

  template <typename E>
  double Evaluate(E& parameters);

  template <typename E, typename G>
  void Gradient(E& parameters,
                G& gradient);

  size_t numFunctions();

  xt::xarray<double> GetInitialPoints();

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
