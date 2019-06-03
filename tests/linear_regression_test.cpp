
#include <fml.hpp>

using namespace fml;
using namespace regression;

int main(){

  arma::mat dataset;
  dataset.load("data/linreg.csv");

  std::cout << dataset;

  return 0;
}