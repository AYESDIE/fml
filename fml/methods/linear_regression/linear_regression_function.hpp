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

  // TODO: THIS PART CAN BE IMPROVED.
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
    score = score.t() - labels;
    gradient = dataset * score / dataset.n_cols;
  }
  else
  {
    // TODO: THIS PART CAN BE IMPROVED.
    arma::mat data = arma::ones<arma::mat>(dataset.n_rows + 1, dataset.n_cols);
    data.submat(0, 0, dataset.n_rows - 1, dataset.n_cols - 1) = dataset;

    gradient = data * (parameters * data  - labels.t()).t() / data.n_cols;
  }
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
    score = score.t() - labels.row(id);
    gradient = dataset.col(id) * score;
  }
  else
  {
    arma::mat data = arma::ones<arma::mat>(dataset.n_rows + 1, dataset.n_cols);
    data.submat(0, 0, dataset.n_rows - 1, dataset.n_cols - 1) = dataset;
    score = parameters * data.col(id);

    score = score.t() - labels.row(id);


    gradient = data.col(id) * score;
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
