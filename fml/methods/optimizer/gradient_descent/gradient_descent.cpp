//
// Created by ayesdie on 7/6/19.
//

#include "gradient_descent.hpp"

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

template<typename DifferentiableFunctionType>
double GradientDescent::Optimize(DifferentiableFunctionType &function,
                                 xt::xarray<double>& iterate)
{
  // Set the maximum value for the objectives.
  double overallObjective = std::numeric_limits<double>::max();
  double lastObjective = std::numeric_limits<double>::max();

  // Iterate
  for (int i = 1; i < maxIterations; ++i)
  {
    overallObjective = function.Evaluate(iterate);

    // Output the objective function.
    std::cout << "Gradient Descent: iteration " << i << ", objective"
        << overallObjective << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      std::cout << "Gradient Descent: converged to " << overallObjective
           << "; terminating" << " with failure.  Try a smaller step size?"
           << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      std::cout << "Gradient Descent: minimized within tolerance "
           << tolerance << "; " << "terminating optimization." << std::endl;
      return overallObjective;
    }

    xt::xarray<double> gradient;
    function.Gradient(iterate, gradient);

    iterate -= stepSize * gradient;

    lastObjective = overallObjective;
  }

  std::cout << "Gradient Descent: maximum iterations (" << maxIterations
       << ") reached; " << "terminating optimization." << std::endl;
  return overallObjective;
}

}
}
