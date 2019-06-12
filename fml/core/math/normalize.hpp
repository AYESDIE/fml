//
// Created by ayesdie on 9/6/19.
//

#ifndef FML_MATH_NORMALIZE_NORMALIZE_HPP
#define FML_MATH_NORMALIZE_NORMALIZE_HPP

#include "fml/core.hpp"

namespace fml {
namespace math {


template <typename E>
void Normalizer(E& xexpression,
               size_t i)
{
  double min = xt::amin(xexpression, {0})(i);
  double max = xt::amax(xexpression, {0})(i);

  if ((max - min) != 0)
  {
    double mean = xt::mean(xexpression, {0})(i);

    xt::view(xexpression,xt::all(), xt::keep(i)) -= mean;
    xt::view(xexpression,xt::all(), xt::keep(i)) /= (max - min);
    return;
  }

  #ifdef FML_DEBUG_CONSOLE
  std::cout << "Normalize: max - min = 0; Terminating." << std::endl;
  #endif
}

template <typename E>
void Normalize(E& xexpression)
{ /* does nothing */ }

template <typename E,
          typename... I>
void Normalize(E& xexpression,
               size_t index,
               I... indices)
{
  fml::math::Normalizer(xexpression, index);
  fml::math::Normalize(xexpression, indices...);
}

} // namespace math
} // namespace fml

#endif //FML_MATH_NORMALIZE_NORMALIZE_HPP
