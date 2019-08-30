//
// Created by ayesdie on 4/7/19.
//

#include <fml/methods/softmax_regression/softmax_regression_function.hpp>
#include <fml/core/optimizers/gradient_descent/gradient_descent.hpp>
#include <fml/core.hpp>
#include <fml/core/log/log.hpp>
#include <fml/core/manipulate/ground_truth.hpp>
#include "catch.hpp"

TEST_CASE("asdad","[asdada]")
{
  xt::xtensor<double, 2> data = {{1, 2, 3, 4, 5},
                                 {6, 7, 8, 9, 10}};

  fml::log(std::cout, data);

  xt::xtensor<double, 2> parameters = {{0.1, 0.2, 0.3, 0.4, 0.5},
                                       {0.6, 0.7, 0.8, 0.9, 1.0},
                                       {1.1, 1.2, 1.3, 1.4, 1.4}};

  xt::xtensor<size_t, 2> labels = {{2},
                                   {0}};

  auto gTT = fml::manipulate::getGroundTruthMatrix(labels, 3);


  // ---------------------------- HOX
  fml::log(std::cout, parameters);
  auto Wx = xt::exp(xt::linalg::dot(parameters, xt::transpose(data)));
  fml::log(std::cout, Wx);
  auto hox = Wx/xt::sum(Wx, {0});
  fml::log(std::cout, "\nhox\n", hox);
  // -----------------------------




  // ---------------------------- JCost
  fml::log(std::cout, gTT);
  fml::log(std::cout, Wx * xt::transpose(gTT) / xt::sum(Wx, {0}));
  // ----------------------------

  xt::xtensor<double, 2> dataT = xt::transpose(data);
  fml::regression::SoftmaxRegressionFunction<> srf(dataT, labels, 3);
  fml::log(std::cout, srf.Evaluate(parameters));
  // ----------------------------

  std::cout << "\n\n\n";
  std::cout << xt::transpose(gTT);
  std::cout << "\n \n" << hox << "\n";
  // ---------------------------- Gradient
    // 1 - P
    auto p = xt::linalg::dot((xt::transpose(gTT) - hox), data);
    fml::log(std::cout, p);

  xt::xtensor<double, 2> grad;
  srf.Gradient(parameters, grad);
    fml::log(std::cout, "\n\n\n\n\n", grad);
  // ----------------------------

  fml::log(std::cout, "random0");

  fml::optimizer::GradientDescent gd(0.01, 100000, 1e-9);
  xt::xtensor<double, 2> parametersT = xt::transpose(parameters);
  gd.Optimize(srf, parameters);
}