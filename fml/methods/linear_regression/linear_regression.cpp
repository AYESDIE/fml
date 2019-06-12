//
// Created by ayesdie on 7/6/19.
//

#include "linear_regression.hpp"

namespace fml {
namespace regression {
LinearRegression::LinearRegression(const xt::xarray<double> &dataset,
                                   const xt::xarray<double> &labels)
{
  LinearRegressionFunction<const xt::xarray<double>,
      const xt::xarray<double>> lrf(dataset, labels);

  parameters = lrf.GetInitialPoints();

  fml::optimizer::GradientDescent gd(0.01, 100000, 1e-5);
  #ifdef FML_DEBUG_CONSOLE
  std::cout << "Linear Regression: Start" << std::endl;
  #endif

  double overallObjective = gd.Optimize(lrf, parameters);

  #ifdef FML_DEBUG_CONSOLE
  std::cout << "Linear Regression: Stop" << std::endl;
  std::cout << "Linear Regression: Overall objective: "
      << overallObjective << "." << std::endl;
  #endif
}


void LinearRegression::Compute(const xt::xarray<double>& dataset,
                               xt::xarray<double>& labels)
{
  labels = xt::linalg::dot(dataset, parameters);
}

} // namespace regression
} // namespace fml


