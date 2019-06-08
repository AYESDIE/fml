//
// Created by ayesdie on 7/6/19.
//

#include <fml/methods/linear_regression/linear_regression.hpp>
#include "catch.hpp"

using namespace fml;

static void scale(xt::xarray<double>& X,
                  double x_min,
                  double x_max)
{
  return;
}

TEST_CASE("LinearRegression","[LinearRegression]")
{
  std::ifstream in_file;
  in_file.open("data/in.csv");
  auto dataset = xt::load_csv<double>(in_file);
  in_file.close();

  in_file.open("data/res.csv");
  auto result = xt::load_csv<double>(in_file);

  auto labels = xt::view(dataset, xt::all(), xt::keep(2));
  auto data = xt::view(dataset, xt::all(), xt::keep(0, 1));

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

TEST_CASE("LinearRegression2","[LinearRegression]")
{
  auto data = xt::xarray<double>
      {{1, 1},
       {1, 3},
       {1, 4},
       {1, 5},
       {1, 6},
       {1, 8},
       {1, 10}};
  auto labels = xt::xarray<double>
      {{10, 3, 5, 3.5, 7, 9.2, 9}};
  std::cout << "\ndata\n" << data;
  labels = xt::transpose(labels);
  std::cout << "\nlabels\n" << labels;

  fml::regression::LinearRegression lr(data, labels);

  xt::xarray<double> lab;
  lr.Compute(data, lab);
  std::cout << lab;
}