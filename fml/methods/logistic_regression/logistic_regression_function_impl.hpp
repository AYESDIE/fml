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
double LogisticRegressionFunction<DatasetType, LabelsType>::Evaluate(E& parameters)
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
template<typename E, typename G>
void LogisticRegressionFunction<DatasetType, LabelsType>::Gradient(E& parameters,
                                                                   G& gradient)
{
  auto sigmoid = 1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)));
  auto error = sigmoid - labels;
  gradient = xt::linalg::dot(xt::transpose(dataset), error / numFunctions())
      + ((lambda / (2 * numFunctions())) * parameters);

}

template<typename DatasetType, typename LabelsType>
size_t LogisticRegressionFunction<DatasetType, LabelsType>::numFunctions()
{
  return dataset.shape(0);
}

template<typename DatasetType, typename LabelsType>
xt::xarray<double> LogisticRegressionFunction<DatasetType, LabelsType>::GetInitialPoints()
{
  return xt::zeros<xt::xarray<double>>({int(dataset.shape(1)), 1});
}

} // namespace regression
} // namespace fml

#endif //FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP
