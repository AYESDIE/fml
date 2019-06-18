//
// Created by ayesdie on 8/6/19.
//

#ifndef FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
#define FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP

#include "gradient_descent.hpp"

namespace fml {
namespace optimizer {

template<typename DifferentiableFunctionType,
         typename E>
double GradientDescent::Optimize(DifferentiableFunctionType &function,
                                 E& iterate)
{
  // Set the maximum value for the objectives.
  double overallObjective = std::numeric_limits<double>::max();
  double lastObjective = std::numeric_limits<double>::max();

  // Iterate
  for (int i = 1; i < maxIterations; ++i)
  {
    overallObjective = function.Evaluate(iterate);

    #ifdef FML_DEBUG_CONSOLE
    if (i % size_t(maxIterations * 0.1) == 0)
      std::cout << "Gradient Descent: iteration " <<  i  << " / " << maxIterations
          << ", objective " << overallObjective << "." << std::endl;
    #endif

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      #ifdef FML_DEBUG_CONSOLE
      std::cout << "Gradient Descent: converged to " << overallObjective
          << "; terminating" << " with failure.  Try a smaller step size?"
          << std::endl;
      #endif

      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      #ifdef FML_DEBUG_CONSOLE
      std::cout << "Gradient Descent: minimized within tolerance "
          << tolerance << "; " << "terminating optimization." << std::endl;
      #endif

      return overallObjective;
    }

    xt::xtensor<double, 2> gradient;
    function.Gradient(iterate, gradient);

    // Update the iterate values.
    iterate -= stepSize * gradient;

    lastObjective = overallObjective;
  }

  #ifdef FML_DEBUG_CONSOLE
  std::cout << "Gradient Descent: maximum iterations (" << maxIterations
      << ") reached; " << "terminating optimization." << std::endl;
  #endif

  return overallObjective;
}

} // namespace optimizer
} // namespace fml

#endif //FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
