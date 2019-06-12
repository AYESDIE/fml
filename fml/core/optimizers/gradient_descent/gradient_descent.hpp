//
// Created by ayesdie on 7/6/19.
//

#ifndef FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP
#define FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP

#include "fml/core.hpp"

namespace fml {
namespace optimizer {

/**
 * Class for Gradient Descent Optimizer.
 */
class GradientDescent
{
public:
  /**
   * Default constructor for Gradient Descent.
   */
  GradientDescent();

  /**
   * Constructor for Gradient Descent Optimizer.
   *
   * @param stepSize - Step size for optimizer.
   */
  GradientDescent(double stepSize);

  /**
   * Constructor for Gradient Descent Optimizer.
   *
   * @param stepSize - Step size for optimizer.
   * @param maxIterations - Maximum number of iterations.
   * @param tolerance - Tolerance to stop the optimization process.
   */
  GradientDescent(double stepSize,
                  size_t maxIterations,
                  double tolerance);

  /**
   * The main optimization function.
   *
   * @tparam DifferentiableFunctionType - Type of function, must have
   * Gradient(const xt::xarray<double>&, xt::xarray<double>&) and
   * Evaluate(const xt::xarray<double>&)
   *
   * @param function - Function to be optimized.
   * @param iterate - Parameters for which the function is optimized.
   * @return - Overall objective
   */
  template <typename DifferentiableFunctionType,
            typename E>
  double Optimize(DifferentiableFunctionType& function,
                  E& iterate);

private:
  /// Step Size
  double stepSize;

  /// Max Iterations
  size_t maxIterations;

  /// Tolerance
  double tolerance;
};

} // namespace optimizer
} // namespace fml

#include "gradient_descent_impl.hpp"

#endif //FML_OPTIMIZER_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP
