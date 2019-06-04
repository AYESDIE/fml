//
// Created by ayesdie on 4/6/19.
//

#ifndef FML_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#define FML_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP

#include "../../fml.hpp"
#include "linear_regression_function.hpp"
#include "../optimizers/sgd/sgd.hpp"

namespace fml {
namespace regression {

class LinearRegression
{
public:
  LinearRegression();

  LinearRegression(const arma::mat& dataset,
                   const arma::vec& labels,
                   const bool& fitIntercept);

  void Compute(const arma::mat& dataset,
               arma::mat& labels);

private:
  arma::mat parameters;
};

LinearRegression::LinearRegression()
{ /* does nothing */ }

LinearRegression::LinearRegression(const arma::mat& dataset,
                                   const arma::vec& labels,
                                   const bool& fitIntercept)
{
  LinearRegressionFunction lrf(dataset, labels, fitIntercept);
  parameters = lrf.initialParameters();

  fml::optimizer::SGD sgd;
  double convergence = sgd.Optimize(lrf, parameters);
}

void LinearRegression::Compute(const arma::mat& dataset,
                               arma::mat& labels)
{
  labels = dataset * parameters;
}

} // namespace regression
} // namespace fml

#endif //FML_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
