//
// Created by ayesdie on 8/6/19.
//

#ifndef FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
#define FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP

#include "gradient_descent.hpp"

namespace fml {
namespace optimizer {

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

    // Update the iterate values.
    iterate -= stepSize * gradient;

    lastObjective = overallObjective;
  }

  std::cout << "Gradient Descent: maximum iterations (" << maxIterations
            << ") reached; " << "terminating optimization." << std::endl;
  return overallObjective;
}

} // namespace optimizer
} // namespace fml

#endif //FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
