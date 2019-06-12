//
// Created by ayesdie on 10/6/19.
//

#ifndef FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
#define FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP

#include "../../core.hpp"

namespace fml {
namespace regression {

class LogisticRegressionFunction
{
public:
  LogisticRegressionFunction(const xt::xarray<double>& dataset,
                             const xt::xarray<size_t>& labels,
                             const double lambda = 0);

  double Evaluate(const xt::xarray<double>& parameters);

  void Gradient(const xt::xarray<double>& parameters,
                xt::xarray<double>& gradient);

  size_t numFunctions();

  xt::xarray<double> GetInitialPoints();

private:
  /// Dataset
  xt::xarray<double> dataset;

  /// Labels
  xt::xarray<size_t> labels;

  /// L2 Regularization constant
  double lambda;
};

} // namespace regression
} // namespace fml

#endif //FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
