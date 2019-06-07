//
// Created by ayesdie on 7/6/19.
//

#include "linear_regression_function.hpp"

namespace fml {
namespace regression {

LinearRegressionFunction::LinearRegressionFunction(const xt::xarray<double>& dataset,
                                                   const xt::xarray<double>& labels) :
                                                   dataset(dataset),
                                                   labels(labels)
{ /* does nothing here */ }

double LinearRegressionFunction::Evaluate(const xt::xarray<double>& parameters)
{
  return xt::sum(dataset * parameters)() / (2 * numFunctions());
}

void LinearRegressionFunction::Gradient(const xt::xarray<double>& parameters,
                                        xt::xarray<double>& gradient)
{
  gradient = xt::transpose(dataset) *
      ((dataset * parameters) - labels);
}

size_t LinearRegressionFunction::numFunctions()
{
  return dataset.dimension();
}
}
}