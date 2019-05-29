/**
 * @file sgd.hpp
 * @author Ayush Chamoli
 *
 * Implementation of Stochastic Gradient Descent.
 */

#ifndef FML_OPTIMIZERS_SGD_SGD_IMPL_HPP
#define FML_OPTIMIZERS_SGD_SGD_IMPL_HPP

namespace fml {

namespace optimizer {

SGD::SGD(const double stepSize,
         const size_t maxIterations,
         const double tolerance) :
         stepSize(stepSize),
         maxIterations(maxIterations),
         tolerance(tolerance)
{ /* This wont do anything */ }

/**
 * Optimize the function (minimize).
 */
template<typename FunctionType>
double SGD::Optimize(FunctionType &function,
                     arma::mat &iterate)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of Optimization.
  size_t currentFunction = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions ; ++i)
  {
    overallObjective += function.Evaluate(iterate, i);
  }

  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  for (size_t i = 1; i < maxIterations; ++i)
  {
    // If the current iteration is the start of sequence.
    if ((currentFunction % numFunctions) == 0)
    {
      // TODO
    }
  }
}

}
}

#endif //FML_OPTIMIZERS_SGD_SGD_IMPL_HPP
