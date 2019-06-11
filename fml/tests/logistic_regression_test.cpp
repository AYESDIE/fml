//
// Created by ayesdie on 10/6/19.
//

#include <fml/methods/logistic_regression/logistic_regression.hpp>
#include <fml/core/math/normalize.hpp>
#include "catch.hpp"

using namespace fml;

TEST_CASE("Evaluate", "[LogisticRegressionFunction]")
{
  std::ifstream in_file;
  in_file.open("data/logistictest.csv");
  xt::xarray<double> dataset = xt::load_csv<double>(in_file);

  xt::xarray<size_t> labels = xt::view(dataset, xt::all(), xt::keep(3));
  auto data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  fml::regression::LogisticRegression lr(data, labels);
  xt::xarray<size_t> lab;
  auto score = lr.Compute(data, lab);

  auto iter = score.begin();
  auto labeliter = labels.begin();

  size_t correct = 0;
  size_t total = 0;
  for (; iter != score.end(); ++iter, ++labeliter)
  {
    total++;
    if ((*iter <= 0.5) && (*labeliter == 0))
      correct++;
    else if ((*iter > 0.5) && (*labeliter == 1))
      correct++;
  }

  std::cout << correct << " / " << total;
}

TEST_CASE("Evaluate2", "[LogisticRegressionFunction]")
{
  xt::xarray<double> data = {{1, 2, 3, 1},
                             {1, 4, 6, 0},
                             {1, 5, 7, 0},
                             {1, 4, 2, 1}};

  xt::xarray<size_t> labels = xt::view(data, xt::all(), xt::keep(3));
  data = xt::view(data, xt::all(), xt::drop(3));

  fml::regression::LogisticRegressionFunction lrf(data, labels);
  auto params = lrf.GetInitialPoints();
  // REQUIRE(lrf.Evaluate(params) == Approx(6.00085152).margin(1e-5));

  params = {{1, 2, 3}};
}

TEST_CASE("er","[asda]")
{
  std::ifstream in_file;
  in_file.open("data/logistictest.csv");
  xt::xarray<double> dataset = xt::load_csv<double>(in_file);

  xt::xarray<size_t> labels = xt::view(dataset, xt::all(), xt::keep(3));
  auto data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));

  fml::regression::LogisticRegressionFunction lrf(data, labels);

  xt::xarray<double> params = xt::transpose(xt::xarray<double> {{ -25.161272, 0.206233, 0.201470}});
  std::cout << params;
  std::cout << lrf.Evaluate(params);

  xt::xarray<double> grad;
  lrf.Gradient(params, grad);
  std::cout << grad;
}