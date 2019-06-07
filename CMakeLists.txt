cmake_minimum_required(VERSION 3.1)
project(fml)

# Set required standard to C++14.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
# if xtensor was built with xsimd support:
# find_package(xsimd REQUIRED)

add_subdirectory(fml)

set(FML_INCLUDE_DIRS ${FML_INCLUDE_DIRS})
set(FML_LIBRARIES ${FML_LIBRARIES})
set(FML_LIBRARY_DIRS ${FML_LIBRARY_DIRS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)


