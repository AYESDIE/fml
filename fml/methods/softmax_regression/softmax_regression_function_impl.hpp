//
// Created by ayesdie on 4/7/19.
//

#ifndef FML_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_IMPL_HPP
#define FML_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_IMPL_HPP

#include "softmax_regression_function.hpp"

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
//  // Evaluate the loss using sigmoid function
//  auto sigmoid =  1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)));
//  auto error = - (labels * xt::log(sigmoid)) - ((1 - labels) * xt::log(1 - sigmoid));
//  double loss = xt::sum(error / numFunctions())();
//
//  // Evaluate the regularization
//  double reg = (lambda / (2 * numFunctions()))
//      * xt::linalg::dot(xt::transpose(parameters), parameters)();
//
//  return loss + reg;
  std::cout << xt::linalg::dot(dataset, parameters);

  auto scores = xt::linalg::dot(dataset, parameters);
  auto exp_scores = xt::exp(scores);

  auto prob_scores = exp_scores / xt::sum(exp_scores, 1);
  // auto prob_scores = 1;
  std::cout << prob_scores;
  return 0;
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