//
// Created by ayesdie on 8/6/19.
//

#ifndef FML_TEST_UTIL_TEST_FUNCTION_HPP
#define FML_TEST_UTIL_TEST_FUNCTION_HPP

namespace fml {
namespace test {


class TestFunction
{
public:
  TestFunction() { }

  //! Get the starting point.
  xt::xarray<double> GetInitialPoint() const { return xt::transpose(xt::xarray<double>{{1, 3, 2}}); }

  //! Evaluate a function.
  double Evaluate(const xt::xarray<double>& coordinates) const;

  //! Evaluate the gradient of a function.
  void Gradient(const xt::xarray<double>& coordinates, xt::xarray<double>& gradient) const;
};

inline double TestFunction::Evaluate(const xt::xarray<double>& coordinates) const
{
  xt::xarray<double> temp = xt::linalg::dot(xt::transpose(coordinates), coordinates);
  return temp(0, 0);
}

inline void TestFunction::Gradient(const xt::xarray<double>& coordinates,
                                   xt::xarray<double>& gradient) const
{
  gradient = 2 * coordinates;
}


}
}

#endif //FML_TEST_UTIL_TEST_FUNCTION_HPP
