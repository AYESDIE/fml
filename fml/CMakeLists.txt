include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)

# Add fml.hpp to list of sources.
set(FML_SRCS ${FML_SRCS}
        "${CMAKE_CURRENT_SOURCE_DIR}/fml.hpp"
        )

## Recurse into both core/ and methods/.
set(DIRS
    core
    methods
    tests
    )

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

