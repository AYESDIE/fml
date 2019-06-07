// This will be used for testing purpose till we
// don't have a proper build system using CMake.

#include "fml.hpp"

int main()
{
  std::ifstream in_file;
  in_file.open("data/in.csv");

  auto data = xt::load_csv<double>(in_file);
  std::cout << data;
}