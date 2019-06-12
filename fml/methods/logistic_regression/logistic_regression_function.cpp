//
// Created by ayesdie on 10/6/19.
//

#include "logistic_regression_function.hpp"

namespace fml {
namespace regression {

LogisticRegressionFunction::LogisticRegressionFunction(const xt::xarray<double>& dataset,
                                                       const xt::xarray<size_t>& labels,
                                                       const double lambda) :
                                                       dataset(dataset),
                                                       labels(labels),
                                                       lambda(lambda)
{ /* does nothing here */ }

double LogisticRegressionFunction::Evaluate(const xt::xarray<double>& parameters)
{
  // Evaluate the loss using sigmoid function
  auto sigmoid =  1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)));
  auto error = - (labels * xt::log(sigmoid)) - ((1 - labels) * xt::log(1 - sigmoid));
  double loss = xt::sum(error / numFunctions())();

  // Evaluate the regularization
  double reg = (lambda / (2 * numFunctions()))
      * xt::linalg::dot(xt::transpose(parameters), parameters)();

  return loss + reg;
}

void LogisticRegressionFunction::Gradient(const xt::xarray<double>& parameters,
                                          xt::xarray<double>& gradient)
{
  auto sigmoid = 1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)));
  auto error = sigmoid - labels;
  gradient = xt::linalg::dot(xt::transpose(dataset), error / numFunctions())
       + ((lambda / (2 * numFunctions())) * parameters);
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
