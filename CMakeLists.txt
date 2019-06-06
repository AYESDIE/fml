cmake_minimum_required(VERSION 3.1)
project(fml)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
# if xtensor was built with xsimd support:
# find_package(xsimd REQUIRED)

add_subdirectory(fml)