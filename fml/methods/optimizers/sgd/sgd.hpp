/**
 * @file sgd.hpp
 * @author Ayush Chamoli
 *
 * Implementation of Stochastic Gradient Descent.
 */

#ifndef FML_OPTIMIZERS_SGD_SGD_HPP
#define FML_OPTIMIZERS_SGD_SGD_HPP

namespace fml {
namespace optimizer {


class SGD {
public:
  SGD(const double stepSize = 0.01,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5);

  template <typename FunctionType>
  double Optimize(FunctionType& function,
                  arma::mat& iterate);

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

private:
  //! The step size for each example.
  double stepSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;
};

} // namespace optimizer
} // namespace fml

#include "sgd_impl.hpp"

#endif //FML_OPTIMIZERS_SGD_SGD_HPP
