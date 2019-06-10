//
// Created by ayesdie on 10/6/19.
//

#include "logistic_regression_function.hpp"

namespace fml {
namespace regression {

LogisticRegressionFunction::LogisticRegressionFunction(const xt::xarray<double>& dataset,
                                                       const xt::xarray<size_t>& labels) :
                                                       dataset(dataset),
                                                       labels(labels)
{ /* does nothing here */ }

double LogisticRegressionFunction::Evaluate(const xt::xarray<double>& parameters)
{
  return xt::sum(- (labels(0) * xt::log(1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)))))
      - ((1 - labels(0)) * (1 - xt::log(1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)))))))()
      / numFunctions();
}

void LogisticRegressionFunction::Gradient(const xt::xarray<double>& parameters,
                                  xt::xarray<double>& gradient)
{
  gradient = xt::linalg::dot(xt::transpose(dataset), (1
      / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)))) - labels(0));
}

size_t LogisticRegressionFunction::numFunctions()
{
  return dataset.shape(0);
}

xt::xarray<double> LogisticRegressionFunction::GetInitialPoints()
{
  return xt::ones<xt::xarray<double>>({int(dataset.shape(1)), 1});
}


} // namespace regression
} // namespace fml
