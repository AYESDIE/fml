
#include "../fml.hpp"
#include "../methods/linear_regression/linear_regression_function.hpp"
#include "catch.hpp"

using namespace fml;
using namespace fml::regression;

TEST_CASE("LinearRegressionFunctionEvaluateTest","[LinearRegressionFunctionTest]")
{

  arma::mat dataset;
  if (!dataset.load("data/linreg.csv", arma::csv_ascii))
  {
    FAIL("couldn't load data");
    return;
  }

  arma::vec labels = dataset.col(2);

  dataset = dataset.cols(0, 1).t();

  LinearRegressionFunction lrf(dataset, labels, false);

  arma::mat parameters = "1 1";
  double score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(score == lrf.Evaluate(parameters));

  parameters = "12 -32";

  score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(score == lrf.Evaluate(parameters));

  lrf = LinearRegressionFunction(dataset, labels, true);

  parameters = "1 1 1";

  score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(score == lrf.Evaluate(parameters));
}


TEST_CASE("LinearRegressionFunctionGradientTest","[LinearRegressionFunctionTest]")
{

  arma::mat dataset;
  if (!dataset.load("data/linreg.csv", arma::csv_ascii))
  {
    FAIL("couldn't load data");
    return;
  }

  arma::vec labels = dataset.col(2);

  dataset = dataset.cols(0, 1).t();

  LinearRegressionFunction lrf(dataset, labels, false);

  arma::mat parameters = "1 1";
  arma::mat gradient = arma::zeros<arma::mat>(2, 1);

  for (int i = 0; i < dataset.n_cols; ++i)
  {
    arma::mat gengradient;
    lrf.Gradient(parameters, gengradient, i);
    gradient += gengradient;
  }
  gradient /= dataset.n_cols;

  arma::mat grad;
  lrf.Gradient(parameters, grad);
  for (int j = 0; j < parameters.n_cols; ++j)
  {
    REQUIRE(grad(j) == gradient(j));
  }
}