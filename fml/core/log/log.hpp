//
// Created by ayesdie on 28/08/19.
//

#ifndef FML_LOG_LOG_HPP
#define FML_LOG_LOG_HPP

#include "fml/core.hpp"

namespace fml
{
/**
 * Last call for variadic fml::log.
 *
 * @tparam outStream - Output stream type.
 * @param os - Output stream.
 */
template  <typename outStream>
void log(outStream& os)
{
  os << "\n";
}

/**
 * Logs the output.
 *
 * @tparam outStream - Output stream type.
 * @tparam outType - Output type.
 * @tparam Ts - Variadic templates.
 * @param os - Output Stream.
 * @param output - Output.
 * @param args - Variadic parameters.
 */
template <typename outStream,
          typename outType,
          typename... Ts>
void log(outStream& os,
         const outType& output,
         const Ts... args)
{
  #ifdef FML_DEBUG_CONSOLE
  os << output;
  log(os, args...);
  #endif
}

} // namespace fml

#endif //FML_LOG_LOG_HPP
