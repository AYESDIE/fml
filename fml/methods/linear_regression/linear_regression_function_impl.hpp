//
// Created by ayesdie on 12/6/19.
//

#ifndef FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_IMPL_HPP
#define FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_IMPL_HPP

#include "linear_regression_function.hpp"

namespace fml {
namespace regression {

template<typename DatasetType, typename LabelsType>
LinearRegressionFunction<DatasetType, LabelsType>::LinearRegressionFunction(DatasetType& dataset,
                                                                            LabelsType& labels,
                                                                            const double lambda) :
                                                                            dataset(dataset),
                                                                            labels(labels),
                                                                            lambda(lambda)
{ /* does nothing here */ }

template<typename DatasetType, typename LabelsType>
template<typename E>
double LinearRegressionFunction<DatasetType, LabelsType>::Evaluate(const E& parameters)
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
double LinearRegressionFunction<DatasetType, LabelsType>::Evaluate(const E& parameters,
                                                                   const size_t& firstId,
                                                                   const size_t& batchSize)
{
  const size_t lastId = firstId + batchSize;

  auto error = xt::linalg::dot(xt::view(dataset, xt::range(firstId, lastId), xt::all()),
      parameters) - xt::view(labels, xt::range(firstId, lastId), xt::all());

  // Evaluate the regularization
  double reg = (lambda / (2 * numFunctions()))
      * xt::linalg::dot(xt::transpose(parameters), parameters)();

  // Evaluate loss.
  double loss = xt::linalg::dot(xt::transpose(error), error
      / (2 * batchSize))();

  return loss + reg;
}


template<typename DatasetType, typename LabelsType>
template<typename E, typename G>
void LinearRegressionFunction<DatasetType, LabelsType>::Gradient(const E& parameters,
                                                                 G& gradient)
{
  // Evaluates the error between the evaluated values and
  // actual values.
  auto error = xt::linalg::dot(dataset, parameters) - labels;

  // Evaluate the regularized gradient.
  gradient = xt::transpose(xt::linalg::dot(xt::transpose(error)
      / numFunctions(), dataset)) + ((lambda / (2 * numFunctions())) * parameters);
}

template<typename DatasetType, typename LabelsType>
template<typename E, typename G>
void
LinearRegressionFunction<DatasetType, LabelsType>::Gradient(const E& parameters,
                                                            const size_t& firstId,
                                                            G& gradient,
                                                            const size_t& batchSize)
{
  const size_t lastId = firstId + batchSize;

  // Evaluates the error between the evaluated values and
  // actual values.
  auto error = xt::linalg::dot(xt::view(dataset, xt::range(firstId, lastId), xt::all()),
      parameters) - xt::view(labels, xt::range(firstId, lastId), xt::all());

  // Evaluate the regularized gradient.
  gradient = xt::transpose(xt::linalg::dot(xt::transpose(error)
      / numFunctions(), xt::view(dataset, xt::range(firstId, lastId), xt::all())))
      + ((lambda / (2 * numFunctions())) * parameters);
}

template<typename DatasetType, typename LabelsType>
size_t LinearRegressionFunction<DatasetType, LabelsType>::numFunctions()
{
  return dataset.shape(0);
}

template<typename DatasetType, typename LabelsType>
xt::xtensor<double, 2> LinearRegressionFunction<DatasetType, LabelsType>::GetInitialPoints()
{
  return xt::ones<xt::xarray<double>>({int(dataset.shape(1)), 1});
}

} // namespace regression
} // namespace fml

#endif //FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_IMPL_HPP
