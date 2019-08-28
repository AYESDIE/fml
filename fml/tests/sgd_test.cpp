//
// Created by ayesdie on 8/6/19.
//

#include <fml/core/optimizers/sgd/sgd.hpp>
#include <fml/core/optimizers/problems/sgd_test_function.hpp>
#include <fml/core/log/log.hpp>
#include "catch.hpp"

TEST_CASE("SGD", "[SGD]")
{
  fml::test::SGDTestFunction f;

  fml::optimizer::SGD sgd(0.01, 1000, 1e-9, 1);

  fml::log(std::cout, "kek ", 10);
  fml::clog(std::cout);

  auto coordinates = f.GetInitialPoint();
  double result = sgd.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates[0] == Approx(0.0).margin(1e-2));
  REQUIRE(coordinates[1] == Approx(0.0).margin(1e-2));
  REQUIRE(coordinates[2] == Approx(0.0).margin(1e-2));
}