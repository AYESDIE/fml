cmake_minimum_required(VERSION 3.0)
project(fml CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

set(DIRS
    fml)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()


