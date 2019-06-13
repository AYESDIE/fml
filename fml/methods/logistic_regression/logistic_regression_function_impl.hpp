//
// Created by ayesdie on 12/6/19.
//

#ifndef FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP
#define FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP

#include "logistic_regression_function.hpp"

namespace fml {
namespace regression {


template<typename DatasetType, typename LabelsType>
LogisticRegressionFunction<DatasetType, LabelsType>::LogisticRegressionFunction(DatasetType& dataset,
                                                                                LabelsType& labels,
                                                                                const double lambda) :
                                                                                dataset(dataset),
                                                                                labels(labels),
                                                                                lambda(lambda)
{ /* does nothing here */ }

template<typename DatasetType, typename LabelsType>
template<typename E>
double LogisticRegressionFunction<DatasetType, LabelsType>::Evaluate(const E& parameters)
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

template<typename DatasetType, typename LabelsType>
template<typename E>
double LogisticRegressionFunction<DatasetType, LabelsType>::Evaluate(const E& parameters,
                                                                     const size_t& firstId,
                                                                     const size_t& batchSize)
{
  const size_t lastId = firstId + batchSize;

  // Evaluate the loss using sigmoid function
  auto sigmoid =  1 / (1 + xt::exp(-xt::linalg::dot(xt::view(dataset, xt::range(firstId, lastId), xt::all()),
      parameters)));
  auto error = - (xt::view(labels, xt::range(firstId, lastId), xt::all()) * xt::log(sigmoid)) -
      ((1 - xt::view(labels, xt::range(firstId, lastId), xt::all())) * xt::log(1 - sigmoid));
  double loss = xt::sum(error/ batchSize)();

  // Evaluate the regularization
  double reg = (lambda / (2 * numFunctions()))
      * xt::linalg::dot(xt::transpose(parameters), parameters)();

  return loss + reg;
}

template<typename DatasetType, typename LabelsType>
template<typename E, typename G>
void LogisticRegressionFunction<DatasetType, LabelsType>::Gradient(const E& parameters,
                                                                   G& gradient)
{
  auto sigmoid = 1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)));
  auto error = sigmoid - labels;
  gradient = xt::transpose(xt::linalg::dot(xt::transpose(error) / numFunctions(), dataset))
      + ((lambda / (2 * numFunctions())) * parameters);
}

template<typename DatasetType, typename LabelsType>
template<typename E, typename G>
void
LogisticRegressionFunction<DatasetType, LabelsType>::Gradient(const E& parameters,
                                                              const size_t& firstId,
                                                              G& gradient,
                                                              const size_t& batchSize)
{
  const size_t lastId = firstId + batchSize;

  auto sigmoid = 1 / (1 + xt::exp(-xt::linalg::dot(xt::view(dataset, xt::range(firstId, lastId), xt::all()), parameters)));
  auto error = sigmoid - xt::view(labels, xt::range(firstId, lastId), xt::all());
  gradient = xt::transpose(xt::linalg::dot(xt::transpose(error) / numFunctions(),
      xt::view(dataset, xt::range(firstId, lastId), xt::all())))
      + ((lambda / (2 * numFunctions())) * parameters);
}

template<typename DatasetType, typename LabelsType>
size_t LogisticRegressionFunction<DatasetType, LabelsType>::numFunctions()
{
  return dataset.shape(0);
}

template<typename DatasetType, typename LabelsType>
xt::xtensor<double, 2> LogisticRegressionFunction<DatasetType, LabelsType>::GetInitialPoints()
{
  return xt::zeros<xt::xarray<double>>({int(dataset.shape(1)), 1});
}

} // namespace regression
} // namespace fml

#endif //FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP
