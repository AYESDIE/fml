//
// Created by ayesdie on 18/6/19.
//

#ifndef FML_CORE_OPTIMIZERS_SGD_SGD_IMPL_HPP
#define FML_CORE_OPTIMIZERS_SGD_SGD_IMPL_HPP

#include "fml/core/log/log.hpp"
#include "sgd.hpp"

namespace fml {
namespace optimizer {

template<typename DifferentiableFunctionType, typename E>
double SGD::Optimize(DifferentiableFunctionType &function, E &iterate)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.numFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;
  double lastObjective = std::numeric_limits<double>::max();

  // Calculate the first objective.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i, batchSize);

  // Iterate
  for (size_t i = 1; i < maxIterations; ++i, ++currentFunction)
  {
    
      if ((i % (maxIterations / 10)) == 0)
        fml::log(std::cout, "SGD: iteration ", i, ", objective ",
            overallObjective, ".");

    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      if (overallObjective != overallObjective)
      {
        fml::log(std::cout, "SGD: converged to ", overallObjective, "; terminating",
            " with failure.  Try a smaller step size?");
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        fml::log(std::cout, "SGD: minimized within tolerance ", tolerance, "; ",
            "terminating optimization.");
        return overallObjective;
      }

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;
    }

    E gradient;
    function.Gradient(iterate, currentFunction, gradient, batchSize);

    // Update the iterate values.
    iterate -= stepSize * gradient;

    // Now add that to the overall objective function.
    overallObjective += function.Evaluate(iterate, currentFunction, batchSize);
  }

  fml::log(std::cout, "SGD: maximum iterations (", maxIterations, ") reached; ",
      "terminating optimization.");

  return overallObjective;
}

} // namespace optimizer
} // namespace fml

#endif //FML_CORE_OPTIMIZERS_SGD_SGD_IMPL_HPP
