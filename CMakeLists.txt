# fml CMake configuration. This project has no configurable options.
cmake_minimum_required(VERSION 2.8.10)
project(fml C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# Ensure we have C++11 features.  Since we support CMake < 3.1, this needs a
# little bit of special handling.
if ((${CMAKE_MAJOR_VERSION} LESS 3 OR
(${CMAKE_MAJOR_VERSION} EQUAL 3 AND ${CMAKE_MINOR_VERSION} LESS 1))
        AND NOT FORCE_CXX11)
    # Older versions of CMake do not support target_compile_features(), so we have
    # to use something kind of hacky.
    include(CMake/CXX11.cmake)
    check_for_cxx11_compiler(HAS_CXX11)
    if(NOT HAS_CXX11)
        message(FATAL_ERROR "No C++11 compiler available!")
    endif()
    enable_cxx11()
else()
    # set required standard to c++11
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

# Include the dependency.
find_package(Armadillo 6.500.0 REQUIRED)
include_directories(BEFORE "${ARMADILLO_INCLUDE_DIR}")
include_directories(BEFORE "${CMAKE_SOURCE_DIR}/include/")

# Install the headers to the correct location.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/fml_bits"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
        PATTERN "*~" EXCLUDE
        PATTERN "*.sw*" EXCLUDE)
install(FILES ${CMAKE_SOURCE_DIR}/include/fml.hpp
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")

enable_testing()
add_subdirectory(tests)