//
// Created by ayesdie on 7/6/19.
//

#include <fml/methods/linear_regression/linear_regression.hpp>
#include <fml/methods/optimizers/gradient_descent/gradient_descent.hpp>

#include "catch.hpp"

using namespace fml;
using namespace fml::optimizer;

TEST_CASE("SimpleSGDTestFunction","[SGDTest]")
{
  fml::optimizer::GradientDescent gd;

  std::ifstream in_file;
  in_file.open("data/in.csv");
  auto data = xt::load_csv<double>(in_file);
  std::cout << data;

  REQUIRE(true);
} 