//
// Created by ayesdie on 10/6/19.
//

#include <fml/methods/logistic_regression/logistic_regression.hpp>
#include <fml/core/optimizers/sgd/sgd.hpp>
#include <fml/core.hpp>
#include "catch.hpp"

using namespace fml;
using namespace fml::regression;

TEST_CASE("LogisticRegressionFunctionSimpleEvaluate", "[LogisticRegressionFunction]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xtensor<size_t, 2> labels =
      xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  LogisticRegressionFunction<> lrf(data, labels);

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

  xt::xtensor<size_t, 2> labels = xt::view(dataset, xt::all(), xt::keep(3));
  xt::xtensor<double, 2> data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  LogisticRegressionFunction<> lrf(data, labels);

  xt::xarray<double> parameters = lrf.GetInitialPoints();
  REQUIRE(lrf.Evaluate(parameters) == Approx(0.693147).margin(1e-5));

  // Value of parameter for which the function is optimized.
  parameters = {{-25.161272, 0.206233, 0.201470}};
  parameters = xt::transpose(parameters);
  REQUIRE(lrf.Evaluate(parameters) == Approx(0.203498).margin(1e-5));
}

TEST_CASE("LogisticRegressionFunctionSeparableEvaluate", "[LogisticRegressionFunction]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xtensor<size_t, 2> labels =
      xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  LogisticRegressionFunction<> lrf(data, labels);

  xt::xarray<double> parameters;

  // These values were calculated by hand.
  parameters = {{1, 1, 1}};
  parameters = xt::transpose(parameters);
  double evaluatedLoss = 0;
  for (int i = 0; i < 4; ++i)
  {
    evaluatedLoss += lrf.Evaluate(parameters, i, 1);
  }
  evaluatedLoss /= lrf.numFunctions();

  REQUIRE(lrf.Evaluate(parameters) == Approx(evaluatedLoss).margin(1e-3));

  parameters = {{9.382532, 0.88376, -7.615013}};
  parameters = xt::transpose(parameters);
  evaluatedLoss = 0;
  for (int i = 0; i < 4; ++i)
  {
    evaluatedLoss += lrf.Evaluate(parameters, i, 1);
  }
  evaluatedLoss /= lrf.numFunctions();

  REQUIRE(lrf.Evaluate(parameters) == Approx(evaluatedLoss).margin(1e-3));

  // Using the L2 Regularization Parameter.
  const double reg = 15.5;
  lrf = LogisticRegressionFunction<> (data, labels, reg);

  parameters = {{-5, 3, -7}};
  parameters = xt::transpose(parameters);
  evaluatedLoss = 0;
  for (int i = 0; i < 4; ++i)
  {
    evaluatedLoss += lrf.Evaluate(parameters, i, 1);
  }
  evaluatedLoss /= lrf.numFunctions();

  REQUIRE(lrf.Evaluate(parameters) == Approx(evaluatedLoss).margin(1e-3));
}

TEST_CASE("LogisticRegressionFunctionRegularizedEvaluate", "[LogisticRegressionFunction]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xtensor<size_t, 2> labels =
      xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  double smallReg = 0.5;
  double bigReg = 20.5;

  LogisticRegressionFunction<> lrf(data, labels);
  LogisticRegressionFunction<> smallRegLrf(data, labels, smallReg);
  LogisticRegressionFunction<> bigRegLrf(data, labels, bigReg);

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

TEST_CASE("LogisticRegressionFunctionSimpleGradient","[LogisticRegressionFunction]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9},
                             {10, 11, 12}};

  xt::xtensor<size_t, 2> labels =
      xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  LogisticRegressionFunction<> lrf(data, labels);

  xt::xarray<double> parameters;

  // These values were calculated by hand.
  parameters = {{1, 1, 1}};
  parameters = xt::transpose(parameters);

  xt::xarray<double> gradient;
  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == Approx(1.249382).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(1.748763).margin(1e-5));
  REQUIRE(gradient(2, 0) == Approx(2.248145).margin(1e-5));

  parameters = {{9.382532, 0.88376, -7.615013}};
  parameters = xt::transpose(parameters);

  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == Approx(-0.002367).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(-0.00022).margin(1e-5));
  REQUIRE(gradient(2, 0) == Approx(0.001927).margin(1e-5));

  parameters = {{-5, 3, -7}};
  parameters = xt::transpose(parameters);

  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == Approx(-4.25).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(-4.75).margin(1e-5));
  REQUIRE(gradient(2, 0) == Approx(-5.25).margin(1e-5));
}

TEST_CASE("LogisticRegressionFunctionComplexGradient","[LogisticRegressionFunction]")
{
  std::ifstream in_file;
  in_file.open("data/logistictest.csv");
  auto dataset = xt::load_csv<double>(in_file);
  in_file.close();

  xt::xtensor<size_t, 2> labels = xt::view(dataset, xt::all(), xt::keep(3));
  xt::xtensor<double, 2> data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  LogisticRegressionFunction<> lrf(data, labels);
  auto parameters = lrf.GetInitialPoints();

  xt::xarray<double> gradient;
  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == Approx(-0.100000).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(-12.009217).margin(1e-5));
  REQUIRE(gradient(2, 0) == Approx(-11.262842).margin(1e-5));

  // Value of parameter for which the function is optimized.
  parameters = {{-25.161272, 0.206233, 0.201470}};
  parameters = xt::transpose(parameters);

  lrf.Gradient(parameters, gradient);
  REQUIRE(gradient(0, 0) == Approx(0.000003).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(0.000226).margin(1e-5));
  REQUIRE(gradient(2, 0) == Approx(0.000122).margin(1e-5));
}

TEST_CASE("LogisticRegressionFunctionSeparableGradient","[LogisticRegressionFunction]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xtensor<size_t, 2> labels =
      xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  LogisticRegressionFunction<> lrf(data, labels);

  xt::xarray<double> parameters;

  // These values were calculated by hand.
  parameters = {{1, 1, 1}};
  parameters = xt::transpose(parameters);

  xt::xtensor<double, 2> evaluatedGradient =
      xt::zeros<xt::xarray<double>>({3, 1});

  for (int i = 0; i < 4; ++i)
  {
    xt::xtensor<double, 2> temp;
    lrf.Gradient(parameters, i, temp, 1);
    evaluatedGradient += temp;
  }

  xt::xtensor<double, 2> gradient;
  lrf.Gradient(parameters, gradient);

  REQUIRE(gradient(0, 0) == Approx(evaluatedGradient(0, 0)).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(evaluatedGradient(1, 0)).margin(1e-5));
  REQUIRE(gradient(2, 0) == Approx(evaluatedGradient(2, 0)).margin(1e-5));

  // Testing with L2 Regularization Parameter.
  double reg = 15.5;
  lrf = LogisticRegressionFunction<>(data, labels, reg);

  evaluatedGradient = xt::zeros<xt::xarray<double>>({3, 1});

  for (int i = 0; i < 4; ++i)
  {
    xt::xtensor<double, 2> temp;
    lrf.Gradient(parameters, i, temp, 1);
    evaluatedGradient += temp;
  }
  // Remove redundant regularizer.
  evaluatedGradient -= (lrf.numFunctions() - 1) *
      ((reg / (2 * lrf.numFunctions())) * parameters);

  lrf.Gradient(parameters, gradient);

  REQUIRE(gradient(0, 0) == Approx(evaluatedGradient(0, 0)).margin(1e-5));
  REQUIRE(gradient(1, 0) == Approx(evaluatedGradient(1, 0)).margin(1e-5));
  REQUIRE(gradient(2, 0) == Approx(evaluatedGradient(2, 0)).margin(1e-5));
}

TEST_CASE("LogisticRegressionFunctionRegularizedGradient","[LogisticRegressionFunction]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xtensor<size_t, 2> labels =
      xt::transpose(xt::xarray<double>{{1, 2, 3, 4}});

  double smallReg = 0.5;
  double bigReg = 20.5;

  LogisticRegressionFunction<> lrf(data, labels);
  LogisticRegressionFunction<> smallRegLrf(data, labels, smallReg);
  LogisticRegressionFunction<> bigRegLrf(data, labels, bigReg);

  xt::xarray<double> parameters;

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

  parameters = {{9.382532, 0.88376, -7.615013}};
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

  parameters = {{-5, 3, -7}};
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

TEST_CASE("SimpleLogisticRegression", "[LogisticRegression]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xtensor<size_t, 2> labels =
      xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  fml::optimizer::SGD sgd(0.01, 10000, 1e-5, 1);
  LogisticRegression<> lr(data, labels, sgd);

  xt::xtensor<size_t, 2> pred;
  lr.Compute(data, pred);

  auto prediter = pred.begin();
  auto labeliter = labels.begin();

  for (; prediter != pred.end() ; ++prediter, ++labeliter)
  {
    REQUIRE((*prediter)==(*labeliter));
  }
}

TEST_CASE("ComplexLogisticRegression", "[LogisticRegression]")
{
  std::ifstream in_file;
  in_file.open("data/logistictest.csv");
  auto dataset = xt::load_csv<double>(in_file);
  in_file.close();

  xt::xtensor<size_t, 2> labels = xt::view(dataset, xt::all(), xt::keep(3));
  xt::xtensor<double, 2> data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  fml::math::Normalize(data, 0, 1, 2);

  fml::optimizer::GradientDescent gd(0.001, 100000, 1e-9);
  LogisticRegression<> lr(data, labels, gd);

  xt::xtensor<size_t, 2> pred;
  lr.Compute(data, pred);

  size_t total = 0;
  size_t correct = 0;
  auto prediter = pred.begin();
  auto labeliter = labels.begin();

  for (; prediter != pred.end() ; ++prediter, ++labeliter)
  {
    total++;
    if ((*prediter)==(*labeliter))
      correct++;
  }

  REQUIRE((double(correct)/total) >= 0.9);
}