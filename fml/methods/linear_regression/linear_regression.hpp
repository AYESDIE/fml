//
// Created by ayesdie on 7/6/19.
//

#ifndef FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#define FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP

#include "../../core.hpp"

namespace fml {
namespace regression {

class LinearRegression {
public:
  LinearRegression(const xt::xarray<double>& dataset,
                   const xt::xarray<double>& labels);

  void Compute(const xt::xarray<double>& dataset,
               xt::xarray<double>& labels);

private:
  xt::xarray<double> parameters;
};

}
}

#endif //FML_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
