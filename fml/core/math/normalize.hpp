//
// Created by ayesdie on 9/6/19.
//

#ifndef FML_MATH_NORMALIZE_NORMALIZE_HPP
#define FML_MATH_NORMALIZE_NORMALIZE_HPP

#include "../../core.hpp"

namespace fml {
namespace math {

xt::xarray<double> Normalize(xt::xarray<double> xexpression,
                             size_t i)
{
  double min = xt::amin(xexpression, {0})(i);
  double max = xt::amax(xexpression, {0})(i);

  if ((max - min) != 0)
  {
    double mean = xt::mean(xexpression, {0})(i);

    xt::view(xexpression,xt::all(), xt::keep(i)) -= mean;
    xt::view(xexpression,xt::all(), xt::keep(i)) /= (max - min);
    return xt::view(xexpression,xt::all(), xt::keep(i));
  }

  std::cout << "Normalize: max - min = 0; Terminating." << std::endl;
  return xt::view(xexpression,xt::all(), xt::keep(i));
}

xt::xarray<double> Normalize(xt::xarray<double> xexpression,
                             xt::xarray<size_t> indices)
{
  for (auto iter = indices.begin(); iter != indices.end() ; iter++)
  {
    xt::view(xexpression,xt::all(), xt::keep(*iter)) = fml::math::Normalize(xexpression, *iter);
  }

  return xexpression;
}

}
}
#endif //FML_MATH_NORMALIZE_NORMALIZE_HPP
