//
// Created by ayesdie on 7/6/19.
//

#ifndef FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP
#define FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP

#include "../../core.hpp"

namespace fml {
namespace regression {

class LinearRegressionFunction
{
public:
  LinearRegressionFunction(const xt::xarray<double>& dataset,
                           const xt::xarray<double>& labels);

  double Evaluate(const xt::xarray<double>& parameters);

  void Gradient(const xt::xarray<double>& parameters,
                xt::xarray<double>& gradient);

  size_t numFunctions();

  xt::xarray<double> GetInitialPoints();

private:
  xt::xarray<double> dataset;

  xt::xarray<double> labels;
};

}
}

#endif //FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_FUNCTION_HPP
