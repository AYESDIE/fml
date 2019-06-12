//
// Created by ayesdie on 8/6/19.
//

#include <fml/core/optimizers/gradient_descent/gradient_descent.hpp>
#include "catch.hpp"
#include "util/test_function.hpp"

TEST_CASE("GradientDescent", "[GradientDescentTest]")
{
  fml::test::TestFunction f;

  fml::optimizer::GradientDescent gd(0.01, 1000, 1e-9);

  auto coordinates = f.GetInitialPoint();
  double result = gd.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates[0] == Approx(0.0).margin(1e-2));
  REQUIRE(coordinates[1] == Approx(0.0).margin(1e-2));
  REQUIRE(coordinates[2] == Approx(0.0).margin(1e-2));
}