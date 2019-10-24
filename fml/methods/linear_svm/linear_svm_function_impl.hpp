//
// Created by ayesdie on 25/10/19.
//

#ifndef FML_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_IMPL_HPP
#define FML_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_IMPL_HPP

#include "linear_svm_function.hpp"
#include "fml/core/manipulate/ground_truth.hpp"

namespace fml {
namespace svm {

template <typename DatasetType, typename LabelsType>
LinearSVMFunction<DatasetType, LabelsType>::LinearSVMFunction(const DatasetType &dataset,
    const LabelsType &labels,
    const size_t numClasses,
    const double lambda,
    const double delta,
    const bool fitIntercept) :
    dataset(dataset),
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
  groundTruth = fml::manipulate::getGroundTruthMatrix(labels, numClasses);

  std::cout << groundTruth;
}

template <typename DatasetType, typename LabelsType>
template <typename E>
double LinearSVMFunction<DatasetType, LabelsType>::Evaluate(const E &parameters) {
  return 0;
}

template <typename DatasetType, typename LabelsType>
template <typename E, typename G>
void LinearSVMFunction<DatasetType, LabelsType>::Gradient(const E &parameters, G &gradient) {

}
} // namespace svm
} // namespace fml

#endif //FML_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_IMPL_HPP
