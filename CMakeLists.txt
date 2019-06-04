# fml CMake configuration. This project has no configurable options.
cmake_minimum_required(VERSION 3.0)
project(fml C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# Include the dependency.
find_package(Armadillo 6.500.0 REQUIRED)
include_directories(BEFORE "${ARMADILLO_INCLUDE_DIR}")
include_directories(BEFORE "${CMAKE_SOURCE_DIR}/include/")

enable_testing()
add_subdirectory(fml)
