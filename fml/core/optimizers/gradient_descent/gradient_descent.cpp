//
// Created by ayesdie on 7/6/19.
//

#include "gradient_descent.hpp"
#include "gradient_descent_impl.hpp"

namespace fml {
namespace optimizer {
GradientDescent::GradientDescent() {
  stepSize = 0.01;
  maxIterations = 100000;
  tolerance = 1e-5;
}

GradientDescent::GradientDescent(double stepSize) :
                                 stepSize(stepSize)
{
  maxIterations = 100000;
  tolerance = 1e-5;
}

GradientDescent::GradientDescent(double stepSize,
                                 size_t maxIterations,
                                 double tolerance) :
                                 stepSize(stepSize),
                                 maxIterations(maxIterations),
                                 tolerance(tolerance)
{ /* does nothing */ }

} // namespace optimizer
} // namespace fml
