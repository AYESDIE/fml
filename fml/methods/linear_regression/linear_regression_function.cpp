//
// Created by ayesdie on 7/6/19.
//

#include "linear_regression_function.hpp"

namespace fml {
namespace regression {

LinearRegressionFunction::LinearRegressionFunction(const xt::xarray<double>& dataset,
                                                   const xt::xarray<double>& labels,
                                                   const double lambda) :
                                                   dataset(dataset),
                                                   labels(labels),
                                                   lambda(lambda)
{ /* does nothing here */ }

double LinearRegressionFunction::Evaluate(const xt::xarray<double>& parameters)
{
  // Evaluates the error between the evaluated values and
  // actual values.
  auto error = xt::linalg::dot(dataset, parameters) - labels;

  // Evaluate the regularization
  double reg = (lambda / (2 * numFunctions()))
      * xt::linalg::dot(xt::transpose(parameters), parameters)();

  // Evaluate loss.
  double loss = xt::linalg::dot(xt::transpose(error), error
      / (2 * numFunctions()))();

  return loss + reg;
}

void LinearRegressionFunction::Gradient(const xt::xarray<double>& parameters,
                                        xt::xarray<double>& gradient)
{
  // Evaluates the error between the evaluated values and
  // actual values.
  auto error = xt::linalg::dot(dataset, parameters) - labels;
  gradient = xt::linalg::dot(xt::transpose(dataset), error
      / numFunctions());
}

size_t LinearRegressionFunction::numFunctions()
{
  return dataset.shape(0);
}

xt::xarray<double> LinearRegressionFunction::GetInitialPoints()
{
  return xt::ones<xt::xarray<double>>({int(dataset.shape(1)), 1});
}

} // namespace regression
} // namespace fml