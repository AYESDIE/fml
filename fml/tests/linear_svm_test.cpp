//
// Created by ayesdie on 25/10/19.
//

#include "catch.hpp"
#include <fml/methods/linear_svm/linear_svm_function.hpp>

TEST_CASE("LinearSVMFunction", "[LinearSVMFunction]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12}};

  xt::xtensor<size_t, 2> labels =
    xt::transpose(xt::xarray<double>{{0, 0, 1, 1}});

  fml::svm::LinearSVMFunction<> lsf(data, labels, 2, 0.0001, 0.0, false);
}