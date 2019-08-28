//
// Created by ayesdie on 28/08/19.
//

#ifndef FML_LOG_LOG_HPP
#define FML_LOG_LOG_HPP

#include "fml/core.hpp"

namespace fml
{
void log(std::ostream& os)
{
  os << "\n";
}


template <typename outType,
          typename... Ts>
void log(std::ostream& os, outType output, Ts... args)
{
  os << output;
  log(os, args...);
}

void clog(std::ostream& os)
{
  os << "lol";
}

} // namespace fml

#endif //FML_LOG_LOG_HPP
