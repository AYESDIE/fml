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

  bool fitIntercept;
};

LinearRegression::LinearRegression()
{ /* does nothing */ }

LinearRegression::LinearRegression(const arma::mat& dataset,
                                   const arma::vec& labels,
                                   const bool& fitIntercept) :
                                   fitIntercept(fitIntercept)
{
  LinearRegressionFunction lrf(dataset, labels, fitIntercept);
  parameters = lrf.initialParameters();

  fml::optimizer::SGD sgd;
  double convergence = sgd.Optimize(lrf, parameters);
}

void LinearRegression::Compute(const arma::mat& dataset,
                               arma::mat& labels)
{
  std::cout << parameters;
  if (!fitIntercept)
  {
    labels =  (parameters * dataset).t();
  }
  else
  {
    // TODO: THIS CAN BE IMPROVED.
    arma::mat data = arma::ones<arma::mat>(dataset.n_rows + 1, dataset.n_cols);
    data.submat(0, 0, dataset.n_rows - 1, dataset.n_cols - 1) = dataset;

    labels = (parameters * data).t();
  }
}

} // namespace regression
} // namespace fml

#endif //FML_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
