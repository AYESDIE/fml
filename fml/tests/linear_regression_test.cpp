//
// Created by ayesdie on 7/6/19.
//

#include <fml/methods/linear_regression/linear_regression.hpp>
#include <fml/core/math/normalize.hpp>
#include "catch.hpp"

using namespace fml;

TEST_CASE("LinearRegression","[LinearRegression]")
{
  std::ifstream in_file;
  in_file.open("data/in.csv");
  auto dataset = xt::load_csv<double>(in_file);
  in_file.close();

  in_file.open("data/res.csv");
  auto result = xt::load_csv<double>(in_file);

  auto labels = xt::view(dataset, xt::all(), xt::keep(3));
  auto data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  data = fml::math::Normalize(data, 0, 1, 2);
  std::cout << data;

  fml::regression::LinearRegression lr(data, labels);

  xt::xarray<double> predictions;
  lr.Compute(data, predictions);

  auto resultiter = result.begin();
  auto prediter = predictions.begin();
  for (; prediter != predictions.end(); prediter++, resultiter++)
  {
    REQUIRE(*prediter == Approx(*resultiter).margin(1e-5));
  }
}
