//
// Created by ayesdie on 12/6/19.
//

#ifndef FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_IMPL_HPP
#define FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_IMPL_HPP

#include "linear_regression.hpp"

namespace fml {
namespace regression {

template<typename DatasetType, typename LabelsType>
template<typename OptimizerType>
LinearRegression<DatasetType, LabelsType>::LinearRegression(const DatasetType &dataset,
                                                            const LabelsType &labels,
                                                            OptimizerType &optimizer)
{
  LinearRegressionFunction<const DatasetType, const LabelsType> lrf(dataset, labels);

  parameters = lrf.GetInitialPoints();

  fml::log(std::cout, "Linear Regression: Start");

  double overallObjective = optimizer.Optimize(lrf, parameters);

  fml::log(std::cout, "Linear Regression: Stop");
  fml::log(std::cout , "Linear Regression: Overall objective: ",
      overallObjective, ".");
}

template<typename DatasetType, typename LabelsType>
void LinearRegression<DatasetType, LabelsType>::Compute(DatasetType &dataset,
                                                        LabelsType &labels)
{
  labels = xt::linalg::dot(dataset, parameters);
}

}
}

#endif //FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_IMPL_HPP
