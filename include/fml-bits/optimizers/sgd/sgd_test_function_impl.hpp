/**
 * @file sgd_test_function_impl.hpp
 * @author Ayush Chamoli
 *
 * Implementation of Test Function for Stochastic
 * Gradient Descent.
 */
#ifndef FML_OPTIMIZERS_SGD_SGD_TEST_FUNCTION_IMPL_HPP
#define FML_OPTIMIZERS_SGD_SGD_TEST_FUNCTION_IMPL_HPP

namespace fml {
namespace test {
double SGDTestFunction::Evaluate(const arma::mat& coordinates, const size_t i)
const
{
  switch (i)
  {
    case 0:
      return -std::exp(-std::abs(coordinates[0]));

    case 1:
      return std::pow(coordinates[1], 2);

    case 2:
      return std::pow(coordinates[2], 4) + 3 * std::pow(coordinates[2], 2);

    default:
      return 0;
  }
}

void SGDTestFunction::Gradient(const arma::mat& coordinates,
                               const size_t i,
                               arma::mat& gradient) const
{
  gradient.zeros(3);
  switch (i)
  {
    case 0:
      if (coordinates[0] >= 0)
        gradient[0] = std::exp(-coordinates[0]);
      else
        gradient[0] = -std::exp(coordinates[1]);
      break;

    case 1:
      gradient[1] = 2 * coordinates[1];
      break;

    case 2:
      gradient[2] = 4 * std::pow(coordinates[2], 3) + 6 * coordinates[2];
      break;
  }
}


}
}

#endif //FML_OPTIMIZERS_SGD_SGD_TEST_FUNCTION_IMPL_HPP
