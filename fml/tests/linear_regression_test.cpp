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

  LinearRegressionFunction<> lrf(data, labels);

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

  xt::xarray<double> labels = xt::view(dataset, xt::all(), xt::keep(3));
  xt::xarray<double> data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  fml::math::Normalize(data, 1, 2);

  LinearRegressionFunction<> lrf(data, labels);

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

  LinearRegressionFunction<> lrf(data, labels);
  LinearRegressionFunction<> smallRegLrf(data, labels, smallReg);
  LinearRegressionFunction<> bigRegLrf(data, labels, bigReg);

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

TEST_CASE("TemplatizedEvaluate","[LinearRegressionFunction]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xarray<double> labels =
      xt::transpose(xt::xarray<double>{{1, 2, 3, 4}});

  LinearRegressionFunction<xt::xtensor<double,2>> lrf(data, labels);

  xt::xtensor<double, 2> parameters;

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

TEST_CASE("SimpleGradient","[LinearRegressionFunction]")
{
  xt::xarray<double> data = {{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9},
                             {10, 11, 12}};

  xt::xarray<double> labels =
      xt::transpose(xt::xarray<double>{{1, 2, 3, 4}});

  LinearRegressionFunction<> lrf(data, labels);

  xt::xarray<double> parameters;

  // These values were calculated by hand.
  parameters = {{1, 1, 1}};
  parameters = xt::transpose(parameters);

  xt::xarray<double> gradient;
  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == 123.5);
  REQUIRE(gradient(1, 0) == 140.5);
  REQUIRE(gradient(2, 0) == 157.5);

  parameters = {{0, 0, 1./3}};
  parameters = xt::transpose(parameters);

  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == 0);
  REQUIRE(gradient(1, 0) == 0);
  REQUIRE(gradient(2, 0) == 0);

  parameters = {{10., -204.5, 23.5}};
  parameters = xt::transpose(parameters);

  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == -7980.25);
  REQUIRE(gradient(1, 0) == -9080.75);
  REQUIRE(gradient(2, 0) == -10181.25);
}

TEST_CASE("ComplexGradient","[LinearRegressionFunction]")
{
  std::ifstream in_file;
  in_file.open("data/in.csv");
  auto dataset = xt::load_csv<double>(in_file);
  in_file.close();

  xt::xarray<double> labels = xt::view(dataset, xt::all(), xt::keep(3));
  xt::xarray<double> data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  fml::math::Normalize(data, 1, 2);

  LinearRegressionFunction<> lrf(data, labels);
  auto parameters = lrf.GetInitialPoints();

  xt::xarray<double> gradient;
  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == Approx(-340411.659574).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(-22932.097461).margin(1e-5));
  REQUIRE(gradient(2, 0) == Approx(-10296.727488).margin(1e-5));

  // Value of parameter for which the function is optimized.
  parameters = {{ 340412.659574,  504776.75649, -34950.601653}};
  parameters = xt::transpose(parameters);

  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == Approx(-4.681071e-07).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(-2.024553e-02).margin(1e-5));
  REQUIRE(gradient(2, 0) == Approx(2.602012e-02).margin(1e-5));
}

TEST_CASE("RegularizedGradient","[LinearRegressionFunction]")
{
  xt::xarray<double> data = {{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9},
                             {10, 11, 12}};

  xt::xarray<double> labels =
      xt::transpose(xt::xarray<double>{{1, 2, 3, 4}});

  double smallReg = 0.5;
  double bigReg = 20.5;

  LinearRegressionFunction<> lrf(data, labels);
  LinearRegressionFunction<> smallRegLrf(data, labels, smallReg);
  LinearRegressionFunction<> bigRegLrf(data, labels, bigReg);

  xt::xarray<double> parameters;

  // These values were calculated by hand.
  parameters = {{1, 1, 1}};
  parameters = xt::transpose(parameters);
  xt::xarray<double> reg = (smallReg / (2 * smallRegLrf.numFunctions()))
      * parameters;

  xt::xarray<double> gradient;
  lrf.Gradient(parameters, gradient);

  xt::xarray<double> smallGradient;
  smallRegLrf.Gradient(parameters, smallGradient);
  REQUIRE(gradient(0, 0) + reg(0, 0) == Approx(smallGradient(0, 0)).margin(1e-5));
  REQUIRE(gradient(1, 0) + reg(1, 0) == Approx(smallGradient(1, 0)).margin(1e-5));
  REQUIRE(gradient(2, 0) + reg(2, 0) == Approx(smallGradient(2, 0)).margin(1e-5));

  reg = (bigReg / (2 * bigRegLrf.numFunctions()))
      * parameters;

  xt::xarray<double> bigGradient;
  bigRegLrf.Gradient(parameters, bigGradient);
  REQUIRE(gradient(0, 0) + reg(0, 0) == Approx(bigGradient(0, 0)).margin(1e-5));
  REQUIRE(gradient(1, 0) + reg(1, 0) == Approx(bigGradient(1, 0)).margin(1e-5));
  REQUIRE(gradient(2, 0) + reg(2, 0) == Approx(bigGradient(2, 0)).margin(1e-5));

  parameters = {{0, 0, 1./3}};
  parameters = xt::transpose(parameters);
  reg = (smallReg / (2 * smallRegLrf.numFunctions()))
      * parameters;

  lrf.Gradient(parameters, gradient);

  smallRegLrf.Gradient(parameters, smallGradient);
  REQUIRE(gradient(0, 0) + reg(0, 0) == Approx(smallGradient(0, 0)).margin(1e-5));
  REQUIRE(gradient(1, 0) + reg(1, 0) == Approx(smallGradient(1, 0)).margin(1e-5));
  REQUIRE(gradient(2, 0) + reg(2, 0) == Approx(smallGradient(2, 0)).margin(1e-5));

  reg = (bigReg / (2 * bigRegLrf.numFunctions()))
      * parameters;

  bigRegLrf.Gradient(parameters, bigGradient);
  REQUIRE(gradient(0, 0) + reg(0, 0) == Approx(bigGradient(0, 0)).margin(1e-5));
  REQUIRE(gradient(1, 0) + reg(1, 0) == Approx(bigGradient(1, 0)).margin(1e-5));
  REQUIRE(gradient(2, 0) + reg(2, 0) == Approx(bigGradient(2, 0)).margin(1e-5));

  parameters = {{10., -204.5, 23.5}};
  parameters = xt::transpose(parameters);
  reg = (smallReg / (2 * smallRegLrf.numFunctions()))
        * parameters;

  lrf.Gradient(parameters, gradient);

  smallRegLrf.Gradient(parameters, smallGradient);
  REQUIRE(gradient(0, 0) + reg(0, 0) == Approx(smallGradient(0, 0)).margin(1e-5));
  REQUIRE(gradient(1, 0) + reg(1, 0) == Approx(smallGradient(1, 0)).margin(1e-5));
  REQUIRE(gradient(2, 0) + reg(2, 0) == Approx(smallGradient(2, 0)).margin(1e-5));

  reg = (bigReg / (2 * bigRegLrf.numFunctions()))
        * parameters;

  bigRegLrf.Gradient(parameters, bigGradient);
  REQUIRE(gradient(0, 0) + reg(0, 0) == Approx(bigGradient(0, 0)).margin(1e-5));
  REQUIRE(gradient(1, 0) + reg(1, 0) == Approx(bigGradient(1, 0)).margin(1e-5));
  REQUIRE(gradient(2, 0) + reg(2, 0) == Approx(bigGradient(2, 0)).margin(1e-5));
}

TEST_CASE("TemplatizedGradient","[LinearRegressionFunction]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xarray<double> labels =
      xt::transpose(xt::xarray<double>{{1, 2, 3, 4}});

  LinearRegressionFunction<xt::xtensor<double, 2>> lrf(data, labels);

  xt::xarray<double> parameters;

  // These values were calculated by hand.
  parameters = {{1, 1, 1}};
  parameters = xt::transpose(parameters);

  xt::xarray<double> gradient;
  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == 123.5);
  REQUIRE(gradient(1, 0) == 140.5);
  REQUIRE(gradient(2, 0) == 157.5);

  parameters = {{0, 0, 1./3}};
  parameters = xt::transpose(parameters);

  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == 0);
  REQUIRE(gradient(1, 0) == 0);
  REQUIRE(gradient(2, 0) == 0);

  parameters = {{10., -204.5, 23.5}};
  parameters = xt::transpose(parameters);

  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == -7980.25);
  REQUIRE(gradient(1, 0) == -9080.75);
  REQUIRE(gradient(2, 0) == -10181.25);
}

TEST_CASE("SimpleLinearRegression","[LinearRegression]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xtensor<double, 2> labels =
      xt::transpose(xt::xarray<double>{{1, 2, 3, 4}});

  fml::optimizer::GradientDescent gd(0.01, 1000000, 1e-15);
  fml::regression::LinearRegression<xt::xtensor<double, 2>,
      xt::xtensor<double, 2>> lr(data, labels, gd);

  xt::xtensor<double, 2> pred;
  lr.Compute(data, pred);

  auto labeliter = labels.begin();
  auto prediter = pred.begin();

  for (; prediter != pred.end() ; ++labeliter, ++prediter)
  {
    REQUIRE((*labeliter) == Approx(*prediter).margin(1e-5));
  }
}

TEST_CASE("ComplexLinearRegression","[LinearRegression]")
{
  std::ifstream in_file;
  in_file.open("data/in.csv");
  auto dataset = xt::load_csv<double>(in_file);
  in_file.close();

  in_file.open("data/res.csv");
  auto result = xt::load_csv<double>(in_file);

  xt::xarray<double> labels = xt::view(dataset, xt::all(), xt::keep(3));
  xt::xarray<double> data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  fml::math::Normalize(data, 0, 1, 2);

  fml::optimizer::GradientDescent gd;
  fml::regression::LinearRegression<> lr(data, labels, gd);

  xt::xarray<double> predictions;
  lr.Compute(data, predictions);

  auto resultiter = result.begin();
  auto prediter = predictions.begin();
  for (; prediter != predictions.end(); prediter++, resultiter++)
  {
    REQUIRE(*prediter == Approx(*resultiter).margin(1e-5));
  }
}
