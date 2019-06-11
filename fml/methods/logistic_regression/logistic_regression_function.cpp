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
  return xt::sum(- (labels * xt::log(1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)))))
      - ((1 - labels) * xt::log((1 - 1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)))))))()
      / numFunctions();
}

void LogisticRegressionFunction::Gradient(const xt::xarray<double>& parameters,
                                  xt::xarray<double>& gradient)
{
  xt::xarray<double> sigmoid = 1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)));
  xt::xarray<double> error = labels - sigmoid;
  gradient = xt::linalg::dot(xt::transpose(dataset), error) / numFunctions();
}

size_t LogisticRegressionFunction::numFunctions()
{
  return dataset.shape(0);
}

xt::xarray<double> LogisticRegressionFunction::GetInitialPoints()
{
  return xt::zeros<xt::xarray<double>>({int(dataset.shape(1)), 1});
}


} // namespace regression
} // namespace fml
