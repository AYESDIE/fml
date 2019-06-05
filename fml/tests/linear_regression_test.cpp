
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
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "12 14";
  score = (parameters * dataset) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "3.9 -0.64";
  score = (parameters * dataset) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "-92 -0.89";
  score = (parameters * dataset) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-3);
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
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "190 14 -11";
  score = (parameters.cols(0, 1) * dataset + arma::accu(parameters.col(2))) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "-12 4 4.43";
  score = (parameters.cols(0, 1) * dataset + arma::accu(parameters.col(2))) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "0.42 -8.3 2.7";
  score = (parameters.cols(0, 1) * dataset + arma::accu(parameters.col(2))) - labels.t();
  score %= score;
  REQUIRE(std::abs(arma::accu(score) / (2 * dataset.n_cols) - lrf.Evaluate(parameters)) <= 1e-3);
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

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "12 84";
  score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "-65 -4.34";
  score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-3);
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

  arma::mat parameters = "1 1 1";
  double score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "12 84 -53";
  score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-3);

  parameters = "-65 -4.34 -8.9";
  score = 0;
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    score += lrf.Evaluate(parameters, i);
  }
  score /= dataset.n_cols;

  REQUIRE(std::abs(score - lrf.Evaluate(parameters)) <= 1e-3);
}

TEST_CASE("Gradient", "[LinearRegressionFunctionTest]")
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
  arma::mat gradient = dataset * (parameters * dataset  - labels.t()).t() / dataset.n_cols;

  arma::mat evaluatedGradient;
  lrf.Gradient(parameters, evaluatedGradient);

  for (int i = 0; i < evaluatedGradient.n_elem; ++i)
  {
    REQUIRE(std::abs(evaluatedGradient(i) - gradient(i)) <= 1e-3);
  }

  parameters = "42.43 -34.5";
  gradient = dataset * (parameters * dataset  - labels.t()).t() / dataset.n_cols;

  lrf.Gradient(parameters, evaluatedGradient);

  for (int i = 0; i < evaluatedGradient.n_elem; ++i)
  {
    REQUIRE(std::abs(evaluatedGradient(i) - gradient(i)) <= 1e-3);
  }
}

TEST_CASE("GradientIntercept", "[LinearRegressionFunctionTest]")
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

  arma::mat data = arma::ones<arma::mat>(dataset.n_rows + 1, dataset.n_cols);
  data.submat(0, 0, dataset.n_rows - 1, dataset.n_cols - 1) = dataset;

  arma::mat parameters = "1 1 1";
  arma::mat gradient = data * (parameters * data  - labels.t()).t() / data.n_cols;

  arma::mat evaluatedGradient;
  lrf.Gradient(parameters, evaluatedGradient);
  REQUIRE(arma::size(evaluatedGradient) == arma::size(gradient));
  for (int i = 0; i < gradient.n_elem; ++i)
  {
    REQUIRE(std::abs(gradient(i) - evaluatedGradient(i)) <= 1e-3);
  }
}

TEST_CASE("SeparableGradient", "[LinearRegressionFunction]")
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

  arma::mat gradient;

  arma::mat parameters = "1 1";
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    gradient = dataset.col(i) * (parameters * dataset.col(i)  - arma::accu(labels.row(i)));

    arma::mat evaluatedGradient;
    lrf.Gradient(parameters, i, evaluatedGradient);

    REQUIRE(arma::size(gradient) == arma::size(evaluatedGradient));
    for (int j = 0; j < gradient.n_elem; ++j)
    {
      REQUIRE(std::abs(gradient(j) - evaluatedGradient(j)) <= 1e-3);
    }
  }

  parameters = "-342.5 3.764";
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    gradient = dataset.col(i) * (parameters * dataset.col(i)  - arma::accu(labels.row(i)));

    arma::mat evaluatedGradient;
    lrf.Gradient(parameters, i, evaluatedGradient);

    REQUIRE(arma::size(gradient) == arma::size(evaluatedGradient));
    for (int j = 0; j < gradient.n_elem; ++j)
    {
      REQUIRE(std::abs(gradient(j) - evaluatedGradient(j)) <= 1e-3);
    }
  }
}

TEST_CASE("SeparableGradientIntercept", "[LinearRegressionFunctionTest]")
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

  arma::mat data = arma::ones<arma::mat>(dataset.n_rows + 1, dataset.n_cols);
  data.submat(0, 0, dataset.n_rows - 1, dataset.n_cols - 1) = dataset;

  // Testing without intercept parameter.
  LinearRegressionFunction lrf(dataset, labels, true);

  arma::mat gradient;

  arma::mat parameters = "1 1 1";
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    gradient = data.col(i) * (parameters * data.col(i)  - arma::accu(labels.row(i)));

    arma::mat evaluatedGradient;
    lrf.Gradient(parameters, i, evaluatedGradient);

    REQUIRE(arma::size(gradient) == arma::size(evaluatedGradient));
    for (int j = 0; j < gradient.n_elem; ++j)
    {
      REQUIRE(std::abs(gradient(j) - evaluatedGradient(j)) <= 1e-3);
    }
  }

  parameters = "-342.5 3.764 -43.7701";
  for (int i = 0; i < dataset.n_cols; ++i)
  {
    gradient = data.col(i) * (parameters * data.col(i)  - arma::accu(labels.row(i)));

    arma::mat evaluatedGradient;
    lrf.Gradient(parameters, i, evaluatedGradient);

    REQUIRE(arma::size(gradient) == arma::size(evaluatedGradient));
    for (int j = 0; j < gradient.n_elem; ++j)
    {
      REQUIRE(std::abs(gradient(j) - evaluatedGradient(j)) <= 1e-3);
    }
  }
}
