//
// Created by ayesdie on 10/6/19.
//

#ifndef FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
#define FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP

#include "logistic_regression_function.hpp"
#include "../../core.hpp"

namespace fml {
namespace regression {

class LogisticRegression{
public:
  LogisticRegression(const xt::xarray<double>& dataset,
                     const xt::xarray<size_t>& labels);

  xt::xarray<double> Compute(const xt::xarray<double>& dataset,
                             xt::xarray<size_t>& labels);

private:
  xt::xarray<double> parameters;
};

}
}

#endif //FML_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_HPP
