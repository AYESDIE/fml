//
// Created by ayesdie on 7/6/19.
//

#include "../fml.hpp"
#include "../methods/linear_regression/linear_regression.hpp"
#include "../methods/optimizer/gradient_descent/gradient_descent.hpp"

#include "catch.hpp"

using namespace fml;
using namespace fml::optimizer;

TEST_CASE("SimpleSGDTestFunction","[SGDTest]")
{
  fml::optimizer::GradientDescent gd;
  REQUIRE(true);
} 