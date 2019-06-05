
#include "../fml.hpp"
#include "../methods/linear_regression/linear_regression.hpp"
#include "catch.hpp"

using namespace fml;
using namespace fml::regression;

TEST_CASE("Evaluate", "[LinearRegressionFunctionTest]")
{
  // Import the dataset.
  arma::mat dataset;
  if (!dataset.load("data/linreg.csv", arma::csv_ascii))
  {
    FAIL("couldn't load data");
    return;
  }

  // Take the labels column out.
  arma::vec labels = dataset.col(2);

  // Remove the labels column from dataset.
  dataset = dataset.cols(0, 1).t();

  // Testing without intercept parameter.
  LinearRegressionFunction lrf(dataset, labels, false);

  arma::mat parameters = "1 1";
  arma::mat score = (parameters * dataset) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-5);

  parameters = "12 14";
  score = (parameters * dataset) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-5);

  parameters = "3.9 -0.64";
  score = (parameters * dataset) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-5);

  parameters = "-92 -0.89";
  score = (parameters * dataset) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-5);
}

TEST_CASE("InterceptEvaluate", "[LinearRegressionFunctionTest]")
{
  // Import the dataset.
  arma::mat dataset;
  if (!dataset.load("data/linreg.csv", arma::csv_ascii))
  {
    FAIL("couldn't load data");
    return;
  }

  // Take the labels column out.
  arma::vec labels = dataset.col(2);

  // Remove the labels column from dataset.
  dataset = dataset.cols(0, 1).t();


  // Testing with intercept parameter.
  LinearRegressionFunction lrf(dataset, labels, true);

  arma::mat parameters = "1 1 1";
  arma::mat score = (parameters.cols(0, 1) * dataset + arma::accu(parameters.col(2))) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-5);

  parameters = "190 14 -11";
  score = (parameters.cols(0, 1) * dataset + arma::accu(parameters.col(2))) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-5);

  parameters = "-12 4 4.43";
  score = (parameters.cols(0, 1) * dataset + arma::accu(parameters.col(2))) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-5);

  parameters = "0.42 -8.3 2.7";
  score = (parameters.cols(0, 1) * dataset + arma::accu(parameters.col(2))) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-5);
}

TEST_CASE("SeparableEvaluate", "[LinearRegressionFunctionTest]")
{
  // Import the dataset.
  arma::mat dataset;
  if (!dataset.load("data/linreg.csv", arma::csv_ascii))
  {
    FAIL("couldn't load data");
    return;
  }

  // Take the labels column out.
  arma::vec labels = dataset.col(2);

  // Remove the labels column from dataset.
  dataset = dataset.cols(0, 1).t();

  // Testing without intercept parameter.
  LinearRegressionFunction lrf(dataset, labels, false);

  arma::mat parameters = "1 1";
  double score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-5);

  parameters = "12 84";
  score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-5);

  parameters = "-65 -4.34";
  score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-5);
}

TEST_CASE("SeprableEvaluateIntercept", "[LinearRegressionFunctionTest]")
{
  // Import the dataset.
  arma::mat dataset;
  if (!dataset.load("data/linreg.csv", arma::csv_ascii))
  {
    FAIL("couldn't load data");
    return;
  }

  // Take the labels column out.
  arma::vec labels = dataset.col(2);

  // Remove the labels column from dataset.
  dataset = dataset.cols(0, 1).t();


  // Testing without intercept parameter.
  LinearRegressionFunction lrf(dataset, labels, true);

  arma::mat parameters = "0 0 0";
  double score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_rows;

  std::cout << score;
  std::cout << "aa" << lrf.Evaluate(parameters);

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-5);

//  parameters = "8.32 5.23 -1.3204";
//  score = 0;
//  for (int i = 0; i < dataset.n_cols; ++i)
//  {
//    score += lrf.Evaluate(parameters, i);
//  }
//  score /= dataset.n_rows;
//
//  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-5);
//
//  parameters = "-32.61 -234.23 -0.43";
//  score = 0;
//  for (int i = 0; i < dataset.n_cols; ++i)
//  {
//    score += lrf.Evaluate(parameters, i);
//  }
//  score /= dataset.n_rows;
//
//  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-5);
}

TEST_CASE("LinearRegressionTest", "[LinearRegressionFunction]")
{
  arma::mat dataset;
  if (!dataset.load("data/linreg.csv", arma::csv_ascii))
  {
    FAIL("couldn't load data");
    return;
  }

  arma::vec labels = dataset.col(2);

  dataset = dataset.cols(0, 1).t();

  LinearRegression lr(dataset, labels, true);
  arma::vec pred;

  lr.Compute(dataset, pred);

  std::cout << pred;
  REQUIRE(false);
}