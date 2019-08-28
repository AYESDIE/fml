//
// Created by ayesdie on 8/6/19.
//

#ifndef FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
#define FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP

#include "gradient_descent.hpp"
#include "fml/core/log/log.hpp"

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
      fml::log(std::cout, "Gradient Descent: iteration ", i, " / ", maxIterations,
          ", objective ", overallObjective, ".");
    #endif

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      #ifdef FML_DEBUG_CONSOLE
      fml::log(std::cout, "Gradient Descent: converged to ", overallObjective,
          "; terminating", " with failure.  Try a smaller step size?");
      #endif

      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      #ifdef FML_DEBUG_CONSOLE
      fml::log(std::cout, "Gradient Descent: minimized within tolerance ",
          tolerance, "; ", "terminating optimization.");
      #endif

      return overallObjective;
    }

    E gradient;
    function.Gradient(iterate, gradient);

    // Update the iterate values.
    iterate -= stepSize * gradient;

    lastObjective = overallObjective;
  }

  #ifdef FML_DEBUG_CONSOLE
  fml::log(std::cout, "Gradient Descent: maximum iterations (", maxIterations,
      ") reached; ", "terminating optimization.");
  #endif

  return overallObjective;
}

} // namespace optimizer
} // namespace fml

#endif //FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
