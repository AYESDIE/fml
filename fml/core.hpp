#ifndef FML_HPP
#define FML_HPP

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
#include <istream>
#include <fstream>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xfunction.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xindex_view.hpp>

#include <xtensor-blas/xlinalg.hpp>

// Comment this to disable debug console.
#define FML_DEBUG_CONSOLE

#include "fml/core/math/normalize.hpp"
#include "fml/core/manipulate/ground_truth.hpp"

#endif //FML_HPP
