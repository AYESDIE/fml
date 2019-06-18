//
// Created by ayesdie on 18/6/19.
//

#ifndef FML_CORE_OPTIMIZERS_PROBLEMS_GRADIENT_DESCENT_TEST_FUNCTION_HPP
#define FML_CORE_OPTIMIZERS_PROBLEMS_GRADIENT_DESCENT_TEST_FUNCTION_HPP

namespace fml {
namespace test {


class GradientDescentTestFunction
{
public:
  GradientDescentTestFunction() { }

  //! Get the starting point.
  xt::xarray<double> GetInitialPoint() const { return xt::transpose(xt::xtensor<double, 2>{{1, 3, 2}}); }

  //! Evaluate a function.
  double Evaluate(const xt::xtensor<double, 2>& coordinates) const;

  //! Evaluate the gradient of a function.
  void Gradient(const xt::xtensor<double, 2>& coordinates, xt::xtensor<double, 2>& gradient) const;
};

double GradientDescentTestFunction::Evaluate(const xt::xtensor<double, 2>& coordinates) const
{
  xt::xarray<double> temp = xt::linalg::dot(xt::transpose(coordinates), coordinates);
  return temp(0, 0);
}

void GradientDescentTestFunction::Gradient(const xt::xtensor<double, 2>& coordinates,
                                           xt::xtensor<double, 2>& gradient) const
{
  gradient = 2 * coordinates;
}


}
}

#endif //FML_CORE_OPTIMIZERS_PROBLEMS_GRADIENT_DESCENT_TEST_FUNCTION_HPP
