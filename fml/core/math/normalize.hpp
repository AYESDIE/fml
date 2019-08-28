//
// Created by ayesdie on 9/6/19.
//

#ifndef FML_MATH_NORMALIZE_NORMALIZE_HPP
#define FML_MATH_NORMALIZE_NORMALIZE_HPP

#include "fml/core.hpp"

namespace fml {
namespace math {

/**
 * Default Normalize() to catch the last call
 * of the Variadic Normalize().
 */
template <typename E>
void Normalize(E& xexpression)
{ /* does nothing */ }

/**
 * Normalize the xexpression based on the formula
 * A[i] = (A[i] - A.mean)/(A.max - A.min)
 *
 * @tparam E - xexpression
 * @tparam I - Variadic template
 * @param xexpression - xexpression to be normalized.
 * @param index - Index to be normalized.
 * @param indices - Variadic variable.
 */
template <typename E,
          typename... I>
void Normalize(E& xexpression,
               size_t index,
               I... indices)
{
  fml::log(std::cout, "Normalize: index: ", index, ".");

  double min = xt::amin(xexpression, {0})(index);
  double max = xt::amax(xexpression, {0})(index);

  if ((max - min) != 0)
  {
    double mean = xt::mean(xexpression, {0})(index);

    xt::view(xexpression,xt::all(), xt::keep(index)) -= mean;
    xt::view(xexpression,xt::all(), xt::keep(index)) /= (max - min);
  }
  else
    fml::log(std::cout, "Normalize: max - min = 0; Terminating.");

  fml::math::Normalize(xexpression, indices...);
}

} // namespace math
} // namespace fml

#endif //FML_MATH_NORMALIZE_NORMALIZE_HPP
