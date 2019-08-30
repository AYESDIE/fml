//
// Created by ayesdie on 4/7/19.
//

#ifndef FML_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_IMPL_HPP
#define FML_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_IMPL_HPP

#include "softmax_regression_function.hpp"
#include "fml/core/manipulate/ground_truth.hpp"

namespace fml {
namespace regression {

template <typename DatasetType, typename LabelsType>
SoftmaxRegressionFunction<DatasetType, LabelsType>::SoftmaxRegressionFunction(
    DatasetType& dataset,
    LabelsType& labels,
    size_t numClasses,
    double lambda) :
    dataset(dataset),
    labels(labels),
    numClasses(numClasses),
    lambda(lambda)
{
  groundTruth = fml::manipulate::getGroundTruthMatrix(labels, numClasses);
}

template <typename DatasetType, typename LabelsType>
template <typename E>
double SoftmaxRegressionFunction<DatasetType, LabelsType>::Evaluate(const E& parameters)
{
  double logLikelihood, weightDecay, cost;

  auto probabilities = xt::exp(xt::linalg::dot(parameters, dataset));

  logLikelihood = xt::sum(probabilities * xt::transpose(groundTruth)
      / xt::sum(probabilities, {0}))() / numFunctions();

  // Evaluate the regularization
  weightDecay = (lambda / (2 * numFunctions()))
      * xt::linalg::dot(xt::transpose(parameters), parameters)();

  cost = - logLikelihood + weightDecay;

  return cost;
}

template <typename DatasetType, typename LabelsType>
template <typename E, typename G>
void SoftmaxRegressionFunction<DatasetType, LabelsType>::Gradient(const E &parameters, G &gradient)
{
  auto probabilities = xt::exp(xt::linalg::dot(parameters, dataset));

  gradient = xt::linalg::dot((probabilities - xt::transpose(groundTruth)), xt::transpose(dataset)) / numFunctions() +
      lambda * parameters;
}

template<typename DatasetType, typename LabelsType>
size_t SoftmaxRegressionFunction<DatasetType, LabelsType>::numFunctions()
{
  return dataset.shape(0);
}

template<typename DatasetType, typename LabelsType>
xt::xtensor<double, 2> SoftmaxRegressionFunction<DatasetType, LabelsType>::GetInitialPoints()
{
  return xt::ones<xt::xarray<double>>({size_t(dataset.shape(1)), numClasses});
}

} // namespace regression
} // namespace fml

#endif //FML_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_IMPL_HPP