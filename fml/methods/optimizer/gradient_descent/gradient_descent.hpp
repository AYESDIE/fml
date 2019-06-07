//
// Created by ayesdie on 7/6/19.
//

#ifndef FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP
#define FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP

#include "../../../core.hpp"

namespace fml {
namespace optimizer {

class GradientDescent
{
public:
  GradientDescent();

  GradientDescent(double stepSize);

  GradientDescent(double stepSize,
                  size_t maxIterations,
                  double tolerance);

  template <typename DifferentiableFunctionType>
  double Optimize(DifferentiableFunctionType& function,
                  xt::xarray<double>& iterate);

private:
  double stepSize;
  size_t maxIterations;
  double tolerance;
};

}
}

#include "gradient_descent_impl.hpp"

#endif //FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP
