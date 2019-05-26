
#ifndef FML_FML_HPP
#define FML_FML_HPP

// certain compilers are way behind the curve
#if (defined(_MSVC_LANG) && (_MSVC_LANG >= 201402L))
#undef  ARMA_USE_CXX11
  #define ARMA_USE_CXX11
#endif

#include <armadillo>

#if !defined(ARMA_USE_CXX11)
// armadillo automatically enables ARMA_USE_CXX11
  // when a C++11/C++14/C++17/etc compiler is detected
  #error "please enable C++11/C++14 mode in your compiler"
#endif

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <climits>
#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <iostream>
#include <string>
#include <sstream>

#include "fml-bits/linear_svm/linear_svm.hpp"

#endif
