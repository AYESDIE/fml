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
  groundTruth = getGroundTruthMatrix(labels, numClasses);
}

} // namespace regression
} // namespace fml

#endif //FML_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_FUNCTION_IMPL_HPP