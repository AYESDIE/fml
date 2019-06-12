//
// Created by ayesdie on 7/6/19.
//

#include <fml/methods/linear_regression/linear_regression.hpp>
#include <fml/core/math/normalize.hpp>
#include "catch.hpp"

using namespace fml;
using namespace fml::regression;

TEST_CASE("SimpleEvaluate", "[LinearRegressionFunction]")
{
  xt::xarray<double> data = {{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9},
                             {10, 11, 12}};

  xt::xarray<double> labels =
      xt::transpose(xt::xarray<double>{{1, 2, 3, 4}});

  LinearRegressionFunction lrf(data, labels);

  xt::xarray<double> parameters;

  // These values were calculated by hand.
  parameters = {{1, 1, 1}};
  parameters = xt::transpose(parameters);
  REQUIRE(lrf.Evaluate(parameters) == 184.5);

  parameters = {{0, 0, 1./3}};
  parameters = xt::transpose(parameters);
  REQUIRE(lrf.Evaluate(parameters) == 0);

  parameters = {{10., -204.5, 23.5}};
  parameters = xt::transpose(parameters);
  REQUIRE(lrf.Evaluate(parameters) == 770672.625);
}

TEST_CASE("ComplexEvaluate", "[LinearRegressionFunction]")
{
  std::ifstream in_file;
  in_file.open("data/in.csv");
  auto dataset = xt::load_csv<double>(in_file);
  in_file.close();

  auto labels = xt::view(dataset, xt::all(), xt::keep(3));
  auto data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  fml::math::Normalize(data, 1, 2);

  LinearRegressionFunction lrf(data, labels);

  auto parameters = lrf.GetInitialPoints();
  REQUIRE(lrf.Evaluate(parameters)
      == Approx(65591174465.408866).margin(1e-5));

  // Value of parameter for which the function is optimized.
  parameters = {{ 340412.659574,  504776.75649, -34950.601653}};
  parameters = xt::transpose(parameters);
  REQUIRE(lrf.Evaluate(parameters) == Approx(2.04328e+09).margin(1e-5));
}

TEST_CASE("RegularizedEvaluate", "[LinearRegressionFunction]")
{
  xt::xarray<double> data = {{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9},
                             {10, 11, 12}};

  xt::xarray<double> labels =
      xt::transpose(xt::xarray<double>{{1, 2, 3, 4}});

  double smallReg = 0.5;
  double bigReg = 20.5;

  LinearRegressionFunction lrf(data, labels);
  LinearRegressionFunction smallRegLrf(data, labels, smallReg);
  LinearRegressionFunction bigRegLrf(data, labels, bigReg);

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

  // These values were calculated by hand.
  parameters = {{0, 0, 1./3}};
  parameters = xt::transpose(parameters);
  reg = xt::linalg::dot(xt::transpose(parameters), parameters)();

  evalReg = (smallReg / (2 * smallRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
      == Approx(smallRegLrf.Evaluate(parameters)).margin(1e-3));

  evalReg = (bigReg / (2 * bigRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
      == Approx(bigRegLrf.Evaluate(parameters)).margin(1e-3));

  parameters = {{10., -204.5, 23.5}};
  parameters = xt::transpose(parameters);
  reg = xt::linalg::dot(xt::transpose(parameters), parameters)();

  evalReg = (smallReg / (2 * smallRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
          == Approx(smallRegLrf.Evaluate(parameters)).margin(1e-3));

  evalReg = (bigReg / (2 * bigRegLrf.numFunctions())) * reg;
  REQUIRE(lrf.Evaluate(parameters) + evalReg
          == Approx(bigRegLrf.Evaluate(parameters)).margin(1e-3));
}

TEST_CASE("LinearRegression","[LinearRegression]")
{
  std::ifstream in_file;
  in_file.open("data/in.csv");
  auto dataset = xt::load_csv<double>(in_file);
  in_file.close();

  in_file.open("data/res.csv");
  auto result = xt::load_csv<double>(in_file);

  auto labels = xt::view(dataset, xt::all(), xt::keep(3));
  auto data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  fml::math::Normalize(data, 0, 1, 2);
  std::cout << data;

  fml::regression::LinearRegression lr(data, labels);

  xt::xarray<double> predictions;
  lr.Compute(data, predictions);

  auto resultiter = result.begin();
  auto prediter = predictions.begin();
  for (; prediter != predictions.end(); prediter++, resultiter++)
  {
    REQUIRE(*prediter == Approx(*resultiter).margin(1e-5));
  }
}
