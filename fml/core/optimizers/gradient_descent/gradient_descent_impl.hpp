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
  for (size_t i = 1; i < maxIterations; ++i)
  {
    overallObjective = function.Evaluate(iterate);

    if (i % (maxIterations / 10) == 0)
      fml::log(std::cout, "Gradient Descent: iteration ", i, " / ", maxIterations,
          ", objective ", overallObjective, ".");

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      fml::log(std::cout, "Gradient Descent: converged to ", overallObjective,
          "; terminating", " with failure.  Try a smaller step size?");

      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      fml::log(std::cout, "Gradient Descent: minimized within tolerance ",
          tolerance, "; ", "terminating optimization.");

      return overallObjective;
    }

    E gradient;
    function.Gradient(iterate, gradient);

    // Update the iterate values.
    iterate -= stepSize * gradient;

    lastObjective = overallObjective;
  }

  fml::log(std::cout, "Gradient Descent: maximum iterations (", maxIterations,
      ") reached; ", "terminating optimization.");

  return overallObjective;
}

} // namespace optimizer
} // namespace fml

#endif //FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
