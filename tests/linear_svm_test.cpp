/**
 * @file linear_svm_test.hpp
 * @author Ayush Chamoli
 *
 * Testing Linear SVM
 *
 */
#include <fml.hpp>
#include "catch.hpp"

using namespace std;
using namespace alt;

/**
 * Test training of linear svm on a simple dataset using
 * Gradient Descent optimizer
 */
TEST_CASE("LinearSVMGradientDescentSimpleTest","[LinearSVMGradientDescentSimpleTest]")
{
  const size_t numClasses = 2;
  const size_t maxIterations = 10000;
  const double stepSize = 0.01;
  const double tolerance = 1e-5;
  const double lambda = 0.0001;
  const double delta = 1.0;

  // A very simple fake dataset
  arma::mat dataset = "2 0 0;"
  "0 0 0;"
  "0 2 1;"
  "1 0 2;"
  "0 1 0";

  //  Corresponding labels
  arma::Row<size_t> labels = "1 0 1";

  // Create a linear svm object using custom gradient descent optimizer.
  ens::GradientDescent optimizer(stepSize, maxIterations, tolerance);
  LinearSVM<arma::mat> lsvm(dataset, labels, numClasses, lambda,
  delta, false, optimizer);

  // Compare training accuracy to 1.
  const double acc = lsvm.ComputeAccuracy(dataset, labels);
  BOOST_REQUIRE_CLOSE(acc, 1.0, 0.5);
}