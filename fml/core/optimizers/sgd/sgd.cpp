//
// Created by ayesdie on 18/6/19.
//

#include "sgd.hpp"

namespace fml {
namespace optimizer {

SGD::SGD()
{
  stepSize = 0.01;
  maxIterations = 100000;
  tolerance = 1e-5;
  batchSize = 1;
}

SGD::SGD(double stepSize) :
         stepSize(stepSize)
{
  maxIterations = 100000;
  tolerance = 1e-5;
  batchSize = 1;
}

SGD::SGD(double stepSize,
         size_t maxIterations,
         double tolerance,
         size_t batchSize) :
         stepSize(stepSize),
         maxIterations(maxIterations),
         tolerance(tolerance),
         batchSize(batchSize)
{ /* does nothing here */ }

} // namespace optimizer
} // namespace fml
