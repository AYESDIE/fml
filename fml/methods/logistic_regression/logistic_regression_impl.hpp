//
// Created by ayesdie on 12/6/19.
//

#ifndef FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP
#define FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP

namespace fml {
namespace regression {

template<typename DatasetType, typename LabelsType>
template<typename OptimizerType>
LogisticRegression<DatasetType, LabelsType>::LogisticRegression(const DatasetType& dataset,
                                                                const LabelsType& labels,
                                                                OptimizerType& optimizer)
{
  LogisticRegressionFunction<const DatasetType, const LabelsType> lrf(dataset, labels);

  parameters = lrf.GetInitialPoints();

  fml::log(std::cout, "Logistic Regression: Start");

  double overallObjective = optimizer.Optimize(lrf, parameters);

  fml::log(std::cout, "Logistic Regression: Stop");
  fml::log(std::cout, "Logistic Regression: Overall objective: ",
      overallObjective, ".");
}

template<typename DatasetType, typename LabelsType>
void
LogisticRegression<DatasetType, LabelsType>::Compute(const DatasetType& dataset, LabelsType& labels)
{
  xt::xtensor<double, 2> score = 1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)));
  labels = score;

  for (size_t i = 0; i < labels.size(); ++i)
  {
    if (score(i, 0) <= 0.5)
      labels(i, 0) = 0;
    else
      labels(i, 0) = 1;
  }
}

} // namespace regression
} // namespace fml

#endif //FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP
