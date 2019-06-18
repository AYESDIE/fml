//
// Created by ayesdie on 18/6/19.
//

#ifndef FML_CORE_OPTIMIZERS_PROBLEMS_SGD_TEST_FUNCTION_HPP
#define FML_CORE_OPTIMIZERS_PROBLEMS_SGD_TEST_FUNCTION_HPP

#include "fml/core.hpp"

namespace fml {
namespace test {

class SGDTestFunction 
{
public:
  SGDTestFunction() { }

  size_t numFunctions() 
  {
    return 1;
  }

  //! Get the starting point.
  xt::xtensor<double, 2> GetInitialPoint() const 
  { 
    return xt::transpose(xt::xtensor<double, 2>{{1, 3, 2}}); 
  }

  //! Evaluate a function.
  double Evaluate(const xt::xtensor<double, 2>& coordinates,
                  size_t id,
                  size_t batchSize) const;

  //! Evaluate the gradient of a function.
  void Gradient(const xt::xtensor<double, 2>& coordinates, 
                size_t id,
                xt::xtensor<double, 2>& gradient,
                size_t batchSize) const;
};

double SGDTestFunction::Evaluate(const xt::xtensor<double, 2>& coordinates,
                                 size_t id,
                                 size_t batchSize) 
const
{
  xt::xarray<double> temp = xt::linalg::dot(xt::transpose(xt::view(coordinates, xt::keep(id), xt::all())),
      xt::view(coordinates, xt::keep(id), xt::all()));
  return temp(0, 0);
}

  //! Evaluate the gradient of a function.
void SGDTestFunction::Gradient(const xt::xtensor<double, 2>& coordinates, 
                               size_t id,
                               xt::xtensor<double, 2>& gradient,
                               size_t batchSize) 
const
{
  gradient = 2 * coordinates;
}

} // namespace optimizer
} // namespace fml 


#endif //FML_CORE_OPTIMIZERS_PROBLEMS_SGD_TEST_FUNCTION_HPP