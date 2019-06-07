include_directories(..)


add_executable(test
        main.cpp
        fml.hpp
        methods/optimizer/gradient_descent/gradient_descent.cpp
        methods/linear_regression/linear_regression.cpp
        methods/linear_regression/linear_regression_function.cpp)

set(input_csv ${CMAKE_CURRENT_SOURCE_DIR}/fml/data/in.csv)
set(output_csv ${CMAKE_CURRENT_BINARY_DIR}/fml/data/out.csv)

add_custom_command(
        OUTPUT  ${output_csv}
        COMMAND version1 ${input_csv} ${output_csv}
        DEPENDS version1 ${input_csv}
)
