//
// Created by ayesdie on 7/6/19.
//

#include "linear_regression.hpp"
#include "linear_regression_function.hpp"
#include "../../core/optimizers/gradient_descent/gradient_descent.hpp"

namespace fml {
namespace regression {
LinearRegression::LinearRegression(const xt::xarray<double> &dataset,
                                   const xt::xarray<double> &labels)
{
  LinearRegressionFunction lrf(dataset, labels);

  parameters = lrf.GetInitialPoints();

  fml::optimizer::GradientDescent gd(0.01, 5000000, 1e-5);
  std::cout << "Linear Regression: Start" << std::endl;
  double overallObjective = gd.Optimize(lrf, parameters);
  std::cout << "Linear Regression: Stop" << std::endl;
  std::cout << "Linear Regression: Overall objective: "
      << overallObjective << "." << std::endl;
}


void LinearRegression::Compute(const xt::xarray<double>& dataset,
                               xt::xarray<double>& labels)
{
  labels = xt::linalg::dot(dataset, parameters);


}

}
}


