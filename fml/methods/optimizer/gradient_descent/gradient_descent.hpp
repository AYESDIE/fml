//
// Created by ayesdie on 7/6/19.
//

#ifndef FML_OPTIMIZER_GRADIENT_DESCENT_HPP
#define FML_OPTIMIZER_GRADIENT_DESCENT_HPP

#include "fml/fml.hpp"

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

#endif //FML_OPTIMIZER_GRADIENT_DESCENT_HPP
