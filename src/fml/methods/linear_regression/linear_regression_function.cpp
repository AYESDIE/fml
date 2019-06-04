
#include "linear_regression_function.hpp"

namespace fml {
namespace regression {

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