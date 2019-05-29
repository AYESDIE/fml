/**
 * @file sgd_test.cpp
 * @author Ayush Chamoli
 *
 * Test for Stochastic Gradient Descent.
 */
#include <fml.hpp>
#include "catch.hpp"

using namespace fml;
using namespace fml::optimizer;
using namespace fml::test;

TEST_CASE("SimpleSGDTestFunction","[SGDTest]")
{
  SGDTestFunction f;
  SGD s(0.0003, 5000000, 1e-9);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(-1.0).epsilon(0.0005));
  REQUIRE(coordinates[0] == Approx(0.0).margin(1e-3));
  REQUIRE(coordinates[1] == Approx(0.0).margin(1e-7));
  REQUIRE(coordinates[2] == Approx(0.0).margin(1e-7));
}