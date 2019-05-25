//
// Created by ayesdie on 25/5/19.
//

#include <iostream>
#include "fml.hpp"

//#define CATCH_CONFIG_MAIN  // catch.hpp will define main()
#define CATCH_CONFIG_RUNNER  // we will define main()
#include "catch.hpp"

int main(int argc, char** argv)
{
  size_t seed = std::time(NULL);
  srand((unsigned int) seed);

  return Catch::Session().run(argc, argv);
}