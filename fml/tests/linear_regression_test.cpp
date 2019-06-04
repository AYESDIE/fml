
#include "../fml.hpp"
#include "../methods/linear_regression/linear_regression_function.hpp"
#include "catch.hpp"

using namespace fml;
using namespace fml::regression;

TEST_CASE("SimpleLinearRegressionFunctionTest","[LinearRegressionFunctionTest]")
{

  arma::mat dataset;
  if (!dataset.load("data/linreg.csv", arma::csv_ascii))
  {
    FAIL("couldn't load data");
    return;
  }

  std::cout << dataset;

  REQUIRE(true);
}