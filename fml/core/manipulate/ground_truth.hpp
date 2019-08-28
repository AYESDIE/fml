//
// Created by ayesdie on 4/7/19.
//

#ifndef FML_MANIPULATE_GROUND_TRUTH_HPP
#define FML_MANIPULATE_GROUND_TRUTH_HPP

#include "fml/core.hpp"

namespace fml {
namespace manipulate {

template <typename E>
xt::xtensor<size_t, 2> getGroundTruthMatrix(const E& xexpression,
                                            const size_t& numClasses)
{
  xt::xtensor<double, 2> groundTruthMatrix = xt::zeros<double>({xexpression.shape(0), numClasses});

  std::vector<std::array<size_t, 2>> indices;

  size_t index = 0;
  for (auto iter = xexpression.begin(); iter != xexpression.end(); iter += 1, index += 1)
  {
    indices.push_back({index, *iter});
  }

  auto index_view = xt::index_view(groundTruthMatrix, indices);
  index_view = 1;
  return groundTruthMatrix;
}

} // namespace manipulate
} // namespace fml

#endif //FML_MANIPULATE_GROUND_TRUTH_HPP