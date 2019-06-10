//
// Created by ayesdie on 10/6/19.
//

#include <fml/methods/logistic_regression/logistic_regression.hpp>
#include "catch.hpp"

using namespace fml;

TEST_CASE("Evaluate", "[LogisticRegressionFunction]")
{
  std::ifstream in_file;
  in_file.open("data/logistictest.csv");
  xt::xarray<double> dataset = xt::load_csv<double>(in_file);

  xt::xarray<size_t> labels = xt::view(dataset, xt::all(), xt::keep(2));
  auto data = xt::view(dataset, xt::all(), xt::keep(0, 1));

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
