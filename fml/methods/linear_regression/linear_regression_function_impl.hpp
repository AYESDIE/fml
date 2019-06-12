//
// Created by ayesdie on 12/6/19.
//

#ifndef FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_IMPL_HPP
#define FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_IMPL_HPP

#include "linear_regression_function.hpp"

namespace fml {
namespace regression {

template<typename DatasetType, typename LabelsType>
LinearRegressionFunction<DatasetType, LabelsType>::LinearRegressionFunction(DatasetType &dataset,
                                                                            LabelsType &labels,
                                                                            const double lambda) :
                                                                            dataset(dataset),
                                                                            labels(labels),
                                                                            lambda(lambda)
{ /* does nothing here */ }

template<typename DatasetType, typename LabelsType>
template<typename E>
double LinearRegressionFunction<DatasetType, LabelsType>::Evaluate(E &parameters)
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

template<typename DatasetType, typename LabelsType>
template<typename E>
void LinearRegressionFunction<DatasetType, LabelsType>::Gradient(E &parameters,
                                                                 xt::xarray<double> &gradient)
{
  // Evaluates the error between the evaluated values and
  // actual values.
  auto error = xt::linalg::dot(dataset, parameters) - labels;

  // Evaluate the regularized gradient.
  gradient = xt::linalg::dot(xt::transpose(dataset), error
      / numFunctions()) + ((lambda / (2 * numFunctions())) * parameters);
}

template<typename DatasetType, typename LabelsType>
size_t LinearRegressionFunction<DatasetType, LabelsType>::numFunctions()
{
  return dataset.shape(0);
}

template<typename DatasetType, typename LabelsType>
xt::xarray<double> LinearRegressionFunction<DatasetType, LabelsType>::GetInitialPoints()
{
  return xt::ones<xt::xarray<double>>({int(dataset.shape(1)), 1});;
}

}
}


#endif //FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_IMPL_HPP
