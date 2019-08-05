//
// Created by ayesdie on 4/7/19.
//

#include <fml/methods/softmax_regression/softmax_regression_function.hpp>
#include <fml/core.hpp>
#include "catch.hpp"

TEST_CASE("asdad","[asdada]")
{
    std::ifstream in_file;
    in_file.open("data/iris.csv");
    auto dataset = xt::load_csv<double>(in_file);
    in_file.close();

    xt::xtensor<size_t, 2> labels = xt::view(dataset, xt::all(), xt::keep(5));
    xt::xtensor<double, 2> data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2, 3, 4));

    REQUIRE(true);

    fml::regression::SoftmaxRegressionFunction<> srf(data, labels, 3);
    auto params = srf.GetInitialPoints();
    auto loss = srf.Evaluate(params);

    REQUIRE(true);
}