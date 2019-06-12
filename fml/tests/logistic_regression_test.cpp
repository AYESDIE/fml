//
// Created by ayesdie on 10/6/19.
//

#include <fml/methods/logistic_regression/logistic_regression.hpp>
#include <fml/core.hpp>
#include "catch.hpp"

using namespace fml;
using namespace fml::regression;

TEST_CASE("LogisticRegressionFunctionSimpleEvaluate", "[LogisticRegressionFunction]")
{
  xt::xarray<double> data = {{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9},
                             {10, 11, 12}};

  xt::xarray<double> labels =
      xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  LogisticRegressionFunction lrf(data, labels);

  xt::xarray<double> parameters;

  // These values were calculated by hand.
  parameters = {{1, 1, 1}};
  parameters = xt::transpose(parameters);
  REQUIRE(lrf.Evaluate(parameters) == Approx(5.2506).margin(1e-3));

  parameters = {{9.382532, 0.88376, -7.615013}};
  parameters = xt::transpose(parameters);
  REQUIRE(lrf.Evaluate(parameters) == Approx(0.0095).margin(1e-3));

  parameters = {{-5, 3, -7}};
  parameters = xt::transpose(parameters);
  REQUIRE(lrf.Evaluate(parameters) == Approx(43.7500).margin(1e-3));
}

TEST_CASE("LogisticRegressionFunctionComplexEvaluate","[LogisticRegressionFunction]")
{
  std::ifstream in_file;
  in_file.open("data/logistictest.csv");
  auto dataset = xt::load_csv<double>(in_file);
  in_file.close();

  xt::xarray<double> labels = xt::view(dataset, xt::all(), xt::keep(3));
  xt::xarray<double> data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  LogisticRegressionFunction lrf(data, labels);

  xt::xarray<double> parameters = lrf.GetInitialPoints();
  REQUIRE(lrf.Evaluate(parameters) == Approx(0.693147).margin(1e-5));

  // Value of parameter for which the function is optimized.
  parameters = {{-25.161272, 0.206233, 0.201470}};
  parameters = xt::transpose(parameters);
  REQUIRE(lrf.Evaluate(parameters) == Approx(0.203498).margin(1e-5));
}

TEST_CASE("LogisticRegressionFunctionRegularizedEvaluate", "[LinearRegressionFunction]")
{
  xt::xarray<double> data = {{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9},
                             {10, 11, 12}};

  xt::xarray<double> labels =
      xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  double smallReg = 0.5;
  double bigReg = 20.5;

  LogisticRegressionFunction lrf(data, labels);
  LogisticRegressionFunction smallRegLrf(data, labels, smallReg);
  LogisticRegressionFunction bigRegLrf(data, labels, bigReg);

  xt::xarray<double> parameters;

  // These values were calculated by hand.
  parameters = {{1, 1, 1}};
  parameters = xt::transpose(parameters);
  double reg = xt::linalg::dot(xt::transpose(parameters), parameters)();

  double evalReg = (smallReg / (2 * smallRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
      == Approx(smallRegLrf.Evaluate(parameters)).margin(1e-3));

  evalReg = (bigReg / (2 * bigRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
      == Approx(bigRegLrf.Evaluate(parameters)).margin(1e-3));

  parameters = {{9.382532, 0.88376, -7.615013}};
  parameters = xt::transpose(parameters);
  reg = xt::linalg::dot(xt::transpose(parameters), parameters)();

  evalReg = (smallReg / (2 * smallRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
      == Approx(smallRegLrf.Evaluate(parameters)).margin(1e-3));

  evalReg = (bigReg / (2 * bigRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
      == Approx(bigRegLrf.Evaluate(parameters)).margin(1e-3));

  parameters = {{-5, 3, -7}};
  parameters = xt::transpose(parameters);
  reg = xt::linalg::dot(xt::transpose(parameters), parameters)();

  evalReg = (smallReg / (2 * smallRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
      == Approx(smallRegLrf.Evaluate(parameters)).margin(1e-3));

  evalReg = (bigReg / (2 * bigRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
      == Approx(bigRegLrf.Evaluate(parameters)).margin(1e-3));
}

TEST_CASE("Evaluate2", "[LogisticRegressionFunction]")
{
  xt::xarray<double> data = {{1, 2, 3, 1},
                             {1, 4, 6, 0},
                             {1, 5, 7, 0},
                             {1, 4, 2, 1}};

  xt::xarray<size_t> labels = xt::view(data, xt::all(), xt::keep(3));
  data = xt::view(data, xt::all(), xt::drop(3));

  fml::regression::LogisticRegressionFunction lrf(data, labels);
  auto params = lrf.GetInitialPoints();
  // REQUIRE(lrf.Evaluate(params) == Approx(6.00085152).margin(1e-5));

  params = {{1, 2, 3}};
}

TEST_CASE("er","[asda]")
{
  xt::xarray<double> data = {{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9},
                             {10, 11, 12}};

  xt::xarray<double> labels =
      xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  std::cout << "jfsf";
  LogisticRegression lr(data, labels);


}