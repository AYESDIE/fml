include_directories(..) # <fml/[whatever]>

# Add core.hpp to list of sources.
set(${FML_SRCS} ${FML_SRCS} "core.hpp")

set(DIRS
    methods
    tests
    )

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

if (WIN32)
    add_library(fml ${FML_SRCS})
else (WIN32)
    add_library(fml SHARED ${FML_SRCS})
endif (WIN32)

set_target_properties(fml
        PROPERTIES
        VERSION 1.0
        SOVERSION 1
        )

# Collect all header files in the library.
file(GLOB_RECURSE INCLUDE_H_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
file(GLOB_RECURSE INCLUDE_HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)
set(INCLUDE_FILES ${INCLUDE_H_FILES} ${INCLUDE_HPP_FILES})

# Set generated executables to be installed.  Unfortunately they must manually
# be entered...
install(TARGETS fml
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
