set(SOURCES
    logistic_regression_function.hpp
    logistic_regression_function_impl.hpp
    logistic_regression.hpp
    logistic_regression_impl.hpp)

set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(FML_SRCS ${FML_SRCS} ${DIR_SRCS} PARENT_SCOPE)