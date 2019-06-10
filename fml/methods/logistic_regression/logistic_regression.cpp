//
// Created by ayesdie on 10/6/19.
//

#include "logistic_regression.hpp"
#include "../../core/optimizers/gradient_descent/gradient_descent.hpp"

namespace fml {
namespace regression {

LogisticRegression::LogisticRegression(const xt::xarray<double> &dataset,
                                       const xt::xarray<size_t> &labels)
{
  LogisticRegressionFunction lrf(dataset, labels);

  parameters = lrf.GetInitialPoints();

  fml::optimizer::GradientDescent gd(0.01, 5000000, 1e-5);
  std::cout << "Linear Regression: Start" << std::endl;
  double overallObjective = gd.Optimize(lrf, parameters);
  std::cout << "Linear Regression: Stop" << std::endl;
  std::cout << "Linear Regression: Overall objective: "
            << overallObjective << "." << std::endl;
}

xt::xarray<double> LogisticRegression::Compute(const xt::xarray<double> &dataset,
                                               xt::xarray<size_t> &labels)
{
  xt::xarray<double> score = 1 / (1 + xt::exp(-xt::linalg::dot(dataset, parameters)));
  labels = xt::transpose(score);
  for (size_t i = 0; i < labels.size(); ++i)
  {
    if (labels(i, 0) <= 0.5)
      labels(i, 0) = 0;
    else
      labels(i, 0) = 1;
  }
  return score;
}


}
}