set(DIRS
    linear_regression
    logistic_regression)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(FML_SRCS ${FML_SRCS} ${DIR_SRCS} PARENT_SCOPE)