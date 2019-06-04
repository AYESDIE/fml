#ifndef FML_INCLUDE_FML_BITS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP
#define FML_INCLUDE_FML_BITS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP

#include "../../fml.hpp"


namespace fml {
namespace regression {

class LinearRegressionFunction {
public:
  LinearRegressionFunction()
  { /* This does nothing */ }

  LinearRegressionFunction(const arma::mat& dataset,
                           const arma::vec& labels,
                           const bool& fitIntercept);

  double Evaluate(const arma::mat& parameters);

  double Evaluate(const arma::mat& parameters,
                  const size_t id);
private:

  arma::mat dataset;

  arma::vec labels;

  bool fitIntercept;
};

LinearRegressionFunction::LinearRegressionFunction(const arma::mat &dataset,
                                                   const arma::vec &labels,
                                                   const bool& fitIntercept) :
    dataset(dataset),
    labels(labels),
    fitIntercept(fitIntercept)
{ /* Nothing to do here */ }

double LinearRegressionFunction::Evaluate(const arma::mat &parameters) {
  arma::mat score;

  if (!fitIntercept)
  {
    score = parameters * dataset;
  }
  else
  {
    score = parameters.head_cols(dataset.n_rows) * dataset +
        arma::accu(parameters.col(dataset.n_rows));
  }


  score = score.t() - labels;
  score %= score;

  return arma::accu(score);
}

double LinearRegressionFunction::Evaluate(const arma::mat &parameters, const size_t id) {
  // Loss is evaluated as
  // Σ (h(x) - y)
  double score;

  if (!fitIntercept)
  {
    score = arma::accu(parameters * dataset.col(id));
  }
  else
  {
    score = arma::accu(parameters.head_cols(dataset.n_rows) * dataset.col(id) +
        parameters.col(dataset.n_rows));
  }

  return std::pow(score - labels(id), 2);
}

}
}

#endif //FML_INCLUDE_FML_BITS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP
