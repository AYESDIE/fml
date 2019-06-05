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
                  const size_t& id);

  void Gradient(const arma::mat& parameters,
                arma::mat& gradient);

  void Gradient(const arma::mat& parameters,
                const size_t& id,
                arma::mat& gradient);

  arma::mat initialParameters();

  size_t NumFunctions()
  {
    return dataset.n_cols;
  }

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

double LinearRegressionFunction::Evaluate(const arma::mat& parameters)
{
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
  score /= dataset.n_cols * 2;

  return arma::accu(score);
}

double LinearRegressionFunction::Evaluate(const arma::mat& parameters,
                                          const size_t& id)
{
  // Loss is evaluated as
  // (1/m)Σ (h(x) - y)
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

  return score(id) / 2;
}

void LinearRegressionFunction::Gradient(const arma::mat& parameters,
                                        arma::mat& gradient)
{
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

  gradient = dataset * score / dataset.n_cols;
}

void LinearRegressionFunction::Gradient(const arma::mat& parameters,
                                        const size_t& id,
                                        arma::mat& gradient)
{

  arma::mat score;

  gradient.set_size(arma::size(parameters));

  if (!fitIntercept)
  {
    score = parameters * dataset.col(id);
  }
  else
  {
    score = parameters.head_cols(dataset.n_rows) * dataset.col(id) +
        arma::accu(parameters.col(dataset.n_rows));
  }

  score = score.t() - labels.row(id);

  if (fitIntercept)
  {
    gradient.submat(0, 0, parameters.n_rows - 1, parameters.n_cols - 2) = score * dataset.col(id).t();
    gradient.col(parameters.n_cols - 1) = score;
  }
  else
  {
    gradient = score * dataset.col(id).t();
  }

}

arma::mat LinearRegressionFunction::initialParameters()
{
  if (!fitIntercept)
  {
    return arma::randu<arma::mat>(1, dataset.n_rows);
  }
  else
  {
    return arma::randu<arma::mat>(1, dataset.n_rows + 1);
  }
}

}
}

#endif //FML_INCLUDE_FML_BITS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP
