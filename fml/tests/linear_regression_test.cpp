
#include "../fml.hpp"
#include "../methods/linear_regression/linear_regression_function.hpp"
#include "catch.hpp"

using namespace fml;
using namespace fml::regression;

TEST_CASE("LinearRegressionFunctionTest","[LinearRegressionFunctionTest]")
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
  lrf.Evaluate(parameters);

  double score = 0;
  for (int i = 0; i < dataset.n_cols; ++i) {
    score += lrf.Evaluate(parameters, i);
  }

  REQUIRE(score == lrf.Evaluate(parameters));

  parameters = "12 -32";
  lrf.Evaluate(parameters);

  score = 0;
  for (int i = 0; i < dataset.n_cols; ++i) {
    score += lrf.Evaluate(parameters, i);
  }

  REQUIRE(score == lrf.Evaluate(parameters));
}

TEST_CASE("LinearRegressionFunctionTestWithIntercept","[LinearRegressionFunctionTest]") {

  arma::mat dataset;
  if (!dataset.load("data/linreg.csv", arma::csv_ascii)) {
    FAIL("couldn't load data");
    return;
  }

  arma::vec labels = dataset.col(2);

  dataset = dataset.cols(0, 1).t();

  LinearRegressionFunction lrf(dataset, labels, true);

  arma::mat parameters = "1 1 1";
  lrf.Evaluate(parameters);

  double score = 0;
  for (int i = 0; i < dataset.n_cols; ++i) {
    score += lrf.Evaluate(parameters, i);
  }

  REQUIRE(score == lrf.Evaluate(parameters));
}