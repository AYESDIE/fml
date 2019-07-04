//
// Created by ayesdie on 4/7/19.
//

#include <fml/methods/softmax_regression/softmax_regression_function.hpp>
#include <fml/core.hpp>
#include "catch.hpp"

TEST_CASE("asdad","[asdada]")
{
  xt::xtensor<size_t, 2> a1 = xt::ones<size_t>({1, 7});
  a1[0, 3] = 0;
  a1[0, 6] = 0;
  a1[0, 4] = 2;
  a1 = xt::transpose(a1);
  std::cout << a1;
  auto gt = fml::manipulate::getGroundTruthMatrix(a1, 3);
  std::cout << std::endl << gt << std::endl;
  REQUIRE(true);
}