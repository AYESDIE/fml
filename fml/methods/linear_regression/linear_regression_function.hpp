#ifndef FML_INCLUDE_FML_BITS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP
#define FML_INCLUDE_FML_BITS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP

#include "../../fml.hpp"


namespace fml {
namespace regression {

class LinearRegressionFunction {
  LinearRegressionFunction()
  { /* This does nothing */ }

  LinearRegressionFunction(const arma::mat& dataset,
                           const arma::vec& labels);

  double Evaluate(const arma::mat& parameters);

  double Evaluate(const arma::mat& parameters,
                  const size_t id);
private:

  arma::mat dataset;

  arma::vec labels;
};

LinearRegressionFunction::LinearRegressionFunction(const arma::mat &dataset,
                                                   const arma::vec &labels) :
    dataset(dataset),
    labels(labels)
{ /* Nothing to do here */ }

double LinearRegressionFunction::Evaluate(const arma::mat &parameters) {
  arma::mat score;

  score = parameters * dataset;
  std::cout << parameters;

  return 0;
}

double LinearRegressionFunction::Evaluate(const arma::mat &parameters, const size_t id) {
  // Loss is evaluated as
  // Σ (h(x) - y)
  double score;
  score = arma::accu(parameters * dataset.cols(id, 1));
  return std::pow(score - labels(id), 2);
}

}
}

#endif //FML_INCLUDE_FML_BITS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP
