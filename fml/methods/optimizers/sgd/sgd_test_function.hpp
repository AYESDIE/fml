/**
 * @file sgd_test_function.hpp
 * @author Ayush Chamoli
 *
 * Implementation of Test Function for Stochastic
 * Gradient Descent.
 */

#ifndef FML_OPTIMIZERS_SGD_SGD_TEST_FUNCTION_HPP
#define FML_OPTIMIZERS_SGD_SGD_TEST_FUNCTION_HPP

namespace fml {
namespace test {

class SGDTestFunction
{
public:
  //! Nothing to do for the constructor.
  SGDTestFunction() { }

  //! Return 3 (the number of functions).
  size_t NumFunctions() const { return 3; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("6; -45.6; 6.2"); }

  //! Evaluate a function.
  double Evaluate(const arma::mat& coordinates, const size_t i) const;

  //! Evaluate the gradient of a function.
  void Gradient(const arma::mat& coordinates,
                const size_t i,
                arma::mat& gradient) const;
};

} // namespace test
} // namespace fml

#include "sgd_test_function_impl.hpp"

#endif //FML_OPTIMIZERS_SGD_SGD_TEST_FUNCTION_HPP
