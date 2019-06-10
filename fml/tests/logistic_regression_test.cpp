//
// Created by ayesdie on 10/6/19.
//

#include <fml/methods/logistic_regression/logistic_regression_function.hpp>
#include "catch.hpp"

using namespace fml;

TEST_CASE("Evaluate", "[LogisticRegressionFunction]")
{
  std::ifstream in_file;
  in_file.open("data/logistictest.csv");
  xt::xarray<double> dataset = xt::load_csv<double>(in_file);

  xt::xarray<size_t> labels = xt::view(dataset, xt::all(), xt::keep(2));
  auto data = xt::view(dataset, xt::all(), xt::keep(0, 1));


  fml::regression::LogisticRegressionFunction lr(data, labels);

  xt::xarray<double> params = lr.GetInitialPoints();
  xt::xarray<double> grx;
  lr.Gradient(params, grx);
}
