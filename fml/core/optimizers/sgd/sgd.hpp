//
// Created by ayesdie on 18/6/19.
//

#ifndef FML_CORE_OPTIMIZERS_SGD_SGD_HPP
#define FML_CORE_OPTIMIZERS_SGD_SGD_HPP


#include "fml/core.hpp"

namespace fml {
namespace optimizer {

/**
 * Class for Stochastic Gradient Descent.
 */
class SGD
{
public:
  /**
   * Default constructor for Stochastic Gradient Descent.
   */
  SGD();

  /**
   * Constructor for Stochastic Gradient Descent Optimizer.
   *
   * @param stepSize - Step size for optimizer.
   */
  SGD(double stepSize);

  /**
   * Constructor for Stochastic Gradient Descent Optimizer.
   *
   * @param stepSize - Step size for optimizer.
   * @param maxIterations - Maximum number of iterations.
   * @param tolerance - Tolerance to stop the optimization process.
   * @param batchSize - Batch size for optimization.
   */
  SGD(double stepSize,
      size_t maxIterations,
      double tolerance,
      size_t batchSize);

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

  /// Batch Size
  size_t batchSize;
};

} // namespace optimizer
} // namespace fml

#include "sgd_impl.hpp"

#endif //FML_CORE_OPTIMIZERS_SGD_SGD_HPP
